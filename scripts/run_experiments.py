"""Run the experiments without the notebook (Checkpoint 2 phases + the Checkpoint 3 pipeline).

Run from the repo root. On Colab, mount Drive in a notebook cell first, then:

  python scripts/run_experiments.py --phase smoke   # tiny runs: checks the script works
  python scripts/run_experiments.py --phase tune    # learning-rate search, English dev only
  python scripts/run_experiments.py --phase final   # seeds 1-3 at the chosen learning rates
  python scripts/run_experiments.py --phase table   # summary table (mean +- std over seeds)
  python scripts/run_experiments.py --phase all --results-dir DIR   # Checkpoint 3: everything below

--settings picks the training data for `final`: zs (English only), fs100 / fs1000 (English plus
that many Swahili sentences from the fixed sample whose seed equals the training seed).

`--phase all` (Checkpoint 3): smoke + learning-rate search for qlora and adapters only (the other
three methods keep the Checkpoint 2 learning rates in configs/), then all five methods x seeds 1-3
for zs, fs1000 and fs100, then the summary table.

Every phase skips runs whose result file already exists, so it is safe to re-run after a
Colab disconnect. Result files have the same format as the notebook's. Each finished or failed
run appends a line to progress.log in the results folder.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

METHODS = ["full_ft", "head_only", "lora", "qlora", "adapters"]
CP2_METHODS = ["full_ft", "head_only", "lora"]  # learning rates already tuned at Checkpoint 2
NEW_METHODS = ["qlora", "adapters"]  # tuned by --phase all
SEEDS = [1, 2, 3]
TUNE_GRID = {  # grids from the comments in configs/*.yaml; head_only extended with 1e-2 (best value was at the old edge)
    "full_ft": [1e-5, 2e-5, 5e-5],
    "head_only": [5e-4, 1e-3, 5e-3, 1e-2],
    "lora": [1e-4, 5e-4, 1e-3],
    "qlora": [1e-4, 5e-4, 1e-3],
    "adapters": [1e-4, 5e-4, 1e-3],
}
TIE_MARGIN = 0.2  # English-dev F1 points: within this of the best, keep the config default
SMOKE_N = 500

_DATA_CACHE: dict = {}


# ----------------------------------------------------------------------------- paths / helpers
def default_roots() -> tuple[Path, Path]:
    drive = Path("/content/drive/MyDrive/FrozenWeights")
    if drive.parent.exists():  # Colab with Drive mounted
        return drive / "outputs", drive / "results" / "checkpoint1"
    return REPO / "outputs", REPO / "results" / "checkpoint1"


def load_cfg(method: str) -> dict:
    import yaml

    return yaml.safe_load(open(REPO / "configs" / f"{method}.yaml"))


def same_lr(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9)


def dev_f1(rec: dict) -> float:
    return 100 * rec["metrics"]["en_dev"]["f1"]


def tune_name(method: str, lr: float) -> str:
    return f"tune_{method}_lr{lr:g}_seed1"


def final_name(method: str, seed: int, setting: str = "zs") -> str:
    # zero-shot keeps the Checkpoint 2 name so old and new files line up
    return f"{method}_seed{seed}" if setting == "zs" else f"{method}_{setting}_seed{seed}"


# ----------------------------------------------------------------------------- one training run
def get_data(cfg: dict, max_train: int | None, setting: str = "zs", seed: int = 1,
             max_eval: int | None = None) -> dict:
    from transformers import AutoTokenizer

    from frozenweights import data

    ml = cfg["max_length"]
    base_key = (cfg["model_name"], ml, max_eval)
    if base_key not in _DATA_CACHE:
        if not (data.PROCESSED_DIR / "sw" / "test.jsonl").exists():
            subprocess.run([sys.executable, str(REPO / "scripts" / "download_data.py")], check=True)
        en = {s: data.load_split("en", s)[:max_eval] for s in ("dev", "test")}
        sw_test = data.load_split("sw", "test")[:max_eval]
        tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        _DATA_CACHE[base_key] = {
            "sw": {"test": sw_test},
            "tok": tok,
            "evals": {
                "en_dev": data.to_hf_dataset(en["dev"], tok, ml),
                "en_test": data.to_hf_dataset(en["test"], tok, ml),
                "sw_test": data.to_hf_dataset(sw_test, tok, ml),
            },
        }
    base = _DATA_CACHE[base_key]
    # zero-shot training data is the same for every seed; few-shot data depends on the seed's sample
    train_key = (cfg["model_name"], ml, max_train, setting, seed if data.SETTINGS[setting] else None)
    if train_key not in _DATA_CACHE:
        _DATA_CACHE[train_key] = data.to_hf_dataset(data.training_rows(setting, seed, max_train), base["tok"], ml)
    return {**base, "train": _DATA_CACHE[train_key]}


def run_one(method, seed, lr, name, results_dir, output_root, max_train=None, tuning=False, setting="zs",
            max_eval=None):
    """Train one model and write <name>.json and <name>_sw_test_predictions.jsonl."""
    import torch
    import transformers
    from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed

    from frozenweights import cost, data, metrics, models

    cfg = load_cfg(method)
    cfg["learning_rate"] = lr
    run_dir = output_root / name
    set_seed(seed)
    d = get_data(cfg, max_train, setting, seed, max_eval)
    tok = d["tok"]

    model = models.build_model(method, cfg["model_name"], cfg)
    params = cost.count_parameters(model)
    try:  # weights as stored on the GPU (4-bit for qlora); not counting activations/optimizer
        footprint_gb = model.get_memory_footprint() / 1024**3
    except Exception:
        footprint_gb = None
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        group_by_length=True,
        fp16=use_cuda,
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_num_workers=2 if use_cuda else 0,
        use_cpu=not use_cuda,  # only reachable with --allow-cpu; never fall back to Apple MPS
    )
    mem_cb = cost.PeakMemoryCallback()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=d["train"],
        eval_dataset=d["evals"]["en_dev"],
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=metrics.compute_metrics,
        callbacks=[mem_cb],
    )
    cost.reset_peak_memory()
    with cost.Timer() as t:
        train_out = trainer.train()

    evals = [h for h in trainer.state.log_history if "eval_f1" in h]
    best_epoch = float(max(evals, key=lambda h: h["eval_f1"])["epoch"])

    trainer.args.group_by_length = False  # keep eval predictions in dataset order
    results, sw_pred = {}, None
    for split, ds in d["evals"].items():
        out = trainer.predict(ds)
        y_true, y_pred = metrics.align_predictions(out.predictions, out.label_ids)
        results[split] = metrics.ner_scores(y_true, y_pred)
        if split == "sw_test":
            sw_pred = (y_true, y_pred)

    ckpt_mb = cost.save_trainable(trainer.model, run_dir / "trainable.safetensors")
    record = {
        "run": name, "method": method, "seed": seed, "setting": data.SETTING_LABELS[setting], "config": cfg,
        "fewshot_size": data.SETTINGS[setting], "fewshot_sample_seed": seed if data.SETTINGS[setting] else None,
        "train_sentences": len(d["train"]), "max_train": max_train, "max_eval": max_eval, "tuning": tuning,
        "best_epoch_en_dev": best_epoch,
        "cost": {**params, "peak_memory_gb": mem_cb.peak_gb, "train_time_min": t.seconds / 60,
                 "model_footprint_gb": footprint_gb,
                 "checkpoint_mb": ckpt_mb, "device": cost.device_name(), "train_steps": train_out.global_step},
        "metrics": results,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    (results_dir / f"{name}.json").write_text(json.dumps(record, indent=2))
    y_true, y_pred = sw_pred
    with open(results_dir / f"{name}_sw_test_predictions.jsonl", "w", encoding="utf-8") as f:
        for r, g, p in zip(d["sw"]["test"], y_true, y_pred):
            f.write(json.dumps({"tokens": r["tokens"][: len(g)], "gold": g, "pred": p}, ensure_ascii=False) + "\n")

    # Free Drive space and GPU memory. The result files above are what the analysis uses.
    shutil.rmtree(run_dir / "trainer", ignore_errors=True)
    if tuning or ckpt_mb > 100:
        (run_dir / "trainable.safetensors").unlink(missing_ok=True)
    del trainer, model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return record


def log_progress(results_dir: Path, line: str) -> None:
    with open(results_dir / "progress.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")


def safe_run(label, failures, results_dir, **kw):
    start = time.time()
    try:
        rec = run_one(results_dir=results_dir, **kw)
        msg = (f"done {label}: en_dev F1 {dev_f1(rec):.2f} | sw_test F1 {100 * rec['metrics']['sw_test']['f1']:.2f} "
               f"| {(time.time() - start) / 60:.1f} min")
        print(f"  {msg}", flush=True)
        log_progress(results_dir, msg)
    except Exception:  # keep going; report at the end
        failures.append(label)
        with open(results_dir / "failures.log", "a") as f:
            f.write(f"=== {label}\n{traceback.format_exc()}\n")
        print(f"  FAILED {label} (see failures.log)", flush=True)
        log_progress(results_dir, f"FAILED {label}")
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ----------------------------------------------------------------------------- tuning selection
def collect_tuning(results_dir: Path) -> dict:
    """{method: {lr: english-dev F1}} from tune_*.json plus the seed-1 final file, if present."""
    out: dict = {m: {} for m in METHODS}
    for m in METHODS:
        files = list(results_dir.glob(f"tune_{m}_lr*_seed1.json")) + [results_dir / f"{m}_seed1.json"]
        for p in files:
            if p.exists():
                r = json.loads(p.read_text())
                lr = r["config"]["learning_rate"]
                if not any(same_lr(lr, k) for k in out[m]):
                    out[m][lr] = dev_f1(r)
    return out


def choose_lrs(tuning: dict, methods: list[str]) -> dict:
    chosen = {}
    for m in methods:
        scores = tuning[m]
        if not scores:
            raise SystemExit(f"No tuning results for {m}; run --phase tune first.")
        best_lr = max(scores, key=scores.get)
        default = load_cfg(m)["learning_rate"]
        default_key = next((k for k in scores if same_lr(k, default)), None)
        if default_key is not None and scores[default_key] >= scores[best_lr] - TIE_MARGIN:
            best_lr = default_key
        chosen[m] = float(best_lr)
    return chosen


def print_tuning(tuning: dict, chosen: dict | None = None) -> None:
    for m in METHODS:
        for lr in sorted(tuning[m]):
            mark = "  <- chosen" if chosen and m in chosen and same_lr(lr, chosen[m]) else ""
            print(f"  {m:10s} lr {lr:<8g} English-dev F1 {tuning[m][lr]:6.2f}{mark}")


# ----------------------------------------------------------------------------- phases
def phase_smoke(args, output_root, results_dir):
    failures: list = []
    n = args.max_train or SMOKE_N
    for m in args.methods:
        name = f"{m}_seed1_n{n}"
        if (results_dir / f"{name}.json").exists():
            print(f"skip {name} (exists)")
            continue
        print(f"smoke {m} ({n} English sentences)", flush=True)
        safe_run(name, failures, results_dir, method=m, seed=1, lr=load_cfg(m)["learning_rate"], name=name,
                 output_root=output_root, max_train=n, max_eval=args.max_eval)
    return failures


def phase_tune(args, output_root, results_dir):
    failures: list = []
    todo = []
    existing = collect_tuning(results_dir)
    for m in args.methods:
        for lr in TUNE_GRID[m]:
            if not any(same_lr(lr, k) for k in existing[m]):
                todo.append((m, lr))
    print(f"{len(todo)} tuning runs to do", flush=True)
    for i, (m, lr) in enumerate(todo, 1):
        label = tune_name(m, lr)
        print(f"[{i}/{len(todo)}] {label}", flush=True)
        safe_run(label, failures, results_dir, method=m, seed=1, lr=lr, name=label, output_root=output_root,
                 tuning=True, max_train=args.max_train, max_eval=args.max_eval)
    tuning = collect_tuning(results_dir)
    done = [m for m in args.methods if tuning[m]]
    chosen = choose_lrs(tuning, done) if done else None
    print("\nEnglish-dev tuning results:")
    print_tuning(tuning, chosen)
    if chosen:
        prev = {}
        p = results_dir / "chosen_lrs.json"
        if p.exists():
            prev = json.loads(p.read_text()).get("chosen", {})
        prev.update(chosen)
        p.write_text(json.dumps({"chosen": prev, "dev_f1": {m: {f"{k:g}": v for k, v in tuning[m].items()} for m in METHODS},
                                 "tie_margin_f1_points": TIE_MARGIN}, indent=2))
        print(f"\nchosen learning rates saved to {p}: {chosen}")
    return failures


def parse_lr_overrides(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = float(v)
    return out


def phase_final(args, output_root, results_dir):
    failures: list = []
    p = results_dir / "chosen_lrs.json"
    lrs = json.loads(p.read_text())["chosen"] if p.exists() else {}
    lrs.update(parse_lr_overrides(args.lr))
    missing = [m for m in args.methods if m not in lrs]
    if missing:
        raise SystemExit(f"No learning rate for {missing}. Run --phase tune, or pass --lr method=value.")
    todo = []
    for setting, m, s in ((st, m, s) for st in args.settings for m in args.methods for s in args.seeds):
        name = final_name(m, s, setting)
        path = results_dir / f"{name}.json"
        if path.exists():
            old_lr = json.loads(path.read_text())["config"]["learning_rate"]
            if same_lr(old_lr, lrs[m]):
                continue
            for suffix in (".json", "_sw_test_predictions.jsonl"):  # keep, but out of the way
                q = results_dir / f"{name}{suffix}"
                if q.exists():
                    q.rename(results_dir / f"stale_{name}{suffix}")
            print(f"  {name}.json used lr {old_lr:g}, not {lrs[m]:g}: renamed to stale_{name}.json", flush=True)
        todo.append((setting, m, s))
    print(f"{len(todo)} final runs to do (learning rates: {lrs})", flush=True)
    for i, (setting, m, s) in enumerate(todo, 1):
        label = final_name(m, s, setting)
        print(f"[{i}/{len(todo)}] {label} lr {lrs[m]:g}", flush=True)
        safe_run(label, failures, results_dir, method=m, seed=s, lr=lrs[m], name=label, output_root=output_root,
                 setting=setting, max_train=args.max_train, max_eval=args.max_eval)
    return failures


def setting_glob(method: str, setting: str) -> str:
    return final_name(method, 0, setting).replace("seed0", "seed?") + ".json"


def build_table(results_dir: Path, methods: list[str], settings: list[str] = ("zs",)):
    import pandas as pd

    from frozenweights.data import SETTINGS

    rows = []
    for setting in settings:
        for m in methods:
            recs = [json.loads(p.read_text()) for p in sorted(results_dir.glob(setting_glob(m, setting)))]
            if not recs:
                continue

            def col(fn):
                return pd.Series([fn(r) for r in recs], dtype=float)

            def ms(s):
                return f"{s.mean():.2f} ± {s.std():.2f}" if len(s) > 1 else f"{s.mean():.2f}"

            sw = lambda k: 100 * col(lambda r: r["metrics"]["sw_test"][k])
            footprint = [r["cost"].get("model_footprint_gb") for r in recs]
            rows.append({
                "method": m, "setting": setting, "swahili sentences": SETTINGS[setting], "seeds": len(recs),
                "lr": recs[0]["config"]["learning_rate"],
                "Swahili F1": ms(sw("f1")), "Swahili P": ms(sw("precision")), "Swahili R": ms(sw("recall")),
                "PER F1": ms(sw("PER_f1")), "ORG F1": ms(sw("ORG_f1")), "LOC F1": ms(sw("LOC_f1")),
                "ORG P": ms(sw("ORG_precision")), "ORG R": ms(sw("ORG_recall")),
                "English dev F1": ms(100 * col(lambda r: r["metrics"]["en_dev"]["f1"])),
                "English test F1": ms(100 * col(lambda r: r["metrics"]["en_test"]["f1"])),
                "trainable params": recs[0]["cost"]["trainable_params"],
                "trainable %": round(recs[0]["cost"]["trainable_pct"], 4),
                "peak GB": round(col(lambda r: r["cost"]["peak_memory_gb"]).mean(), 2),
                "weights GB": round(float(pd.Series(footprint, dtype=float).mean()), 3) if None not in footprint else None,
                "minutes": round(col(lambda r: r["cost"]["train_time_min"]).mean(), 1),
                "checkpoint MB": round(col(lambda r: r["cost"]["checkpoint_mb"]).mean(), 1),
                "GPU": ", ".join(sorted({r["cost"]["device"] for r in recs})),
            })
    return pd.DataFrame(rows).set_index(["method", "setting"]) if rows else pd.DataFrame()


def phase_table(args, output_root, results_dir):
    settings = args.settings or list(SETTING_ORDER)
    df = build_table(results_dir, args.methods, settings)
    if df.empty:
        print("No final result files found (expected <method>_seed<N>.json or <method>_<setting>_seed<N>.json).")
        return []
    if "zs" in df.index.get_level_values("setting"):  # same file and layout as Checkpoint 2
        df.xs("zs", level="setting").to_csv(results_dir / "summary_zero_shot.csv")
    out = results_dir / "summary_all_settings.csv"
    df.to_csv(out)
    print(df.T.to_string())
    print(f"\nsaved to {out}")
    return []


SETTING_ORDER = ("zs", "fs1000", "fs100")  # +1,000 before +100, as planned at Checkpoint 2


def phase_all(args, output_root, results_dir):
    """Checkpoint 3 pipeline: smoke + tune the new methods, then every method x seed x setting, then the table."""
    failures: list = []
    # Learning rates: keep the Checkpoint 2 choices (configs/) for the three tuned methods.
    p = results_dir / "chosen_lrs.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    rec.setdefault("chosen", {})
    for m in CP2_METHODS:
        rec["chosen"].setdefault(m, float(load_cfg(m)["learning_rate"]))
    p.write_text(json.dumps(rec, indent=2))

    def with_methods(ms, **extra):
        return argparse.Namespace(**{**vars(args), "methods": ms, **extra})

    new = [m for m in args.methods if m in NEW_METHODS]
    log_progress(results_dir, f"phase all: smoke {new}")
    smoke_failed = phase_smoke(with_methods(new), output_root, results_dir)
    broken = {m for m in new if f"{m}_seed1_n{args.max_train or SMOKE_N}" in smoke_failed}
    if broken:
        print(f"smoke test failed for {sorted(broken)}: skipping them for the rest of the run", flush=True)
    failures += smoke_failed

    to_tune = [m for m in new if m not in broken]
    if to_tune:
        log_progress(results_dir, f"phase all: tune {to_tune}")
        failures += phase_tune(with_methods(to_tune), output_root, results_dir)

    chosen = json.loads(p.read_text())["chosen"]
    methods = [m for m in args.methods if m in chosen and m not in broken]
    skipped = [m for m in args.methods if m not in methods]
    if skipped:
        print(f"no learning rate for {skipped}: not running their final runs", flush=True)
    settings = args.settings or list(SETTING_ORDER)
    for setting in settings:  # one setting at a time so a cut-off run still leaves complete settings
        log_progress(results_dir, f"phase all: final {setting} {methods}")
        failures += phase_final(with_methods(methods, settings=[setting]), output_root, results_dir)

    phase_table(with_methods(args.methods, settings=settings), output_root, results_dir)
    log_progress(results_dir, f"phase all: finished, {len(failures)} failure(s)")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["smoke", "tune", "final", "table", "all"])
    ap.add_argument("--methods", nargs="+", default=None, choices=METHODS,
                    help=f"default: {' '.join(CP2_METHODS)} (all five for --phase all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--settings", nargs="+", default=None, choices=list(SETTING_ORDER),
                    help="training data for final/table: zs, fs100, fs1000 (default: zs for final, all for all/table)")
    ap.add_argument("--lr", nargs="*", help="override learning rates, e.g. --lr lora=1e-3 head_only=5e-3")
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--allow-cpu", action="store_true", help="train without a GPU (debugging only: cost numbers are meaningless)")
    ap.add_argument("--max-train", type=int, default=None, help="debug: cap English training sentences")
    ap.add_argument("--max-eval", type=int, default=None, help="debug: cap sentences per evaluation set")
    args = ap.parse_args()
    if args.methods is None:
        args.methods = METHODS if args.phase == "all" else CP2_METHODS
    if args.settings is None and args.phase == "final":
        args.settings = ["zs"]

    os.chdir(REPO)
    out_default, res_default = default_roots()
    output_root = args.output_root or out_default
    results_dir = args.results_dir or res_default
    results_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"results: {results_dir}\noutputs: {output_root}", flush=True)

    if args.phase != "table" and not args.allow_cpu:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU found. In Colab: Runtime > Change runtime type > GPU, then re-run setup. "
                             "(--allow-cpu exists for debugging, but its time/memory numbers must not be reported.)")

    phases = {"smoke": phase_smoke, "tune": phase_tune, "final": phase_final, "table": phase_table, "all": phase_all}
    failures = phases[args.phase](args, output_root, results_dir)
    if failures:
        print(f"\n{len(failures)} run(s) failed: {failures}. Fix the cause (see failures.log) and re-run the same command.")
        sys.exit(1)


if __name__ == "__main__":
    main()
