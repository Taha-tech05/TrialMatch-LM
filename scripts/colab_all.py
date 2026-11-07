"""One-click Colab run: environment log -> merge -> merged VAL eval -> metrics -> KV-cache bench -> attention bench.

  python scripts/colab_all.py                 # full run
  python scripts/colab_all.py --quick         # --quick / fewer repeats in every benchmark
  python scripts/colab_all.py --stop-on-failure --only merge eval_merged_val
  python scripts/colab_all.py --skip bench_attention

Logs: $TRIALMATCH_OUT/logs/<step>.log. Results: $TRIALMATCH_OUT/results/ (repo results/ if the variable is unset).
Summary: <results>/benchmarks/run_summary.json, rewritten after every step.

Failure policy: a benchmark that exits with code 1 (token/numerical MISMATCH) is recorded as "mismatch" and the run
continues. Any other failure in a benchmark is recorded as "failed" and the run continues unless --stop-on-failure.
A failure in a setup step (env, merge, eval, metrics) stops the run unless --keep-going.
The test split is never evaluated: every command is checked against that before it runs.
"""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trialmatch.model.load import default_merged_dir  # noqa: E402
from trialmatch.training.config import load_train_config, output_root  # noqa: E402

STEPS = ["env", "merge", "eval_merged_val", "compute_metrics", "bench_kv_cache", "bench_attention", "bench_speculative"]
BENCHMARKS = {"bench_kv_cache", "bench_attention", "bench_speculative"}
MERGED_TAG = "finetuned_merged"


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n== {text}\n" + "=" * 78, flush=True)


def assert_no_test_split(cmd: list[str]) -> None:
    flat = " ".join(cmd)
    if "--confirm-final" in flat or ("--split" in cmd and cmd[cmd.index("--split") + 1] == "test"):
        raise SystemExit(f"refusing to run a command that touches the test split: {flat}")


def run_cmd(cmd: list[str], log_path: Path) -> int:
    """Run a command, echo its output to the console and to log_path, return the exit code."""
    assert_no_test_split(cmd)
    crashed: list[bool] = []
    print("$ " + " ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            if line.startswith("Traceback (most recent call last)"):
                crashed.append(True)
        code = proc.wait()
    # An uncaught exception also exits with 1; only a clean exit 1 means "benchmark reported a mismatch".
    return 2 if code == 1 and crashed else code


def step_env(log_path: Path, out: Path) -> int:
    """Environment log (GPU, versions, git commit) written to the step log and results/logs/env_colab_all.json."""
    info = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
            "platform": platform.platform(), "trialmatch_out": str(out)}
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        info["git_commit"] = None
    try:
        import torch
        import transformers
        info.update(torch=torch.__version__, transformers=transformers.__version__, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            info.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                        gpu_total_gib=round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2))
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = repr(e)
    if shutil.which("nvidia-smi"):
        info["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    text = json.dumps(info, indent=2)
    print(text)
    log_path.write_text(text, encoding="utf-8")
    jp = out / "results" / "logs" / "env_colab_all.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(text, encoding="utf-8")
    return 0


def merged_dir_ready(d: Path) -> bool:
    """Complete save only: save_merged() writes merge_info.json LAST, after config and weights, so a partial/interrupted
    save (config.json but no weights) is not treated as done and gets re-merged."""
    has_weights = d.is_dir() and any(d.glob("*.safetensors")) or (d / "pytorch_model.bin").exists()
    return d.is_dir() and (d / "config.json").exists() and (d / "merge_info.json").exists() and bool(has_weights)


def sync_val_predictions(out: Path) -> None:
    """compute_metrics.py reads <repo>/results/eval; evaluate.py writes to $TRIALMATCH_OUT/results/eval. Copy VAL files."""
    src, dst = out / "results" / "eval", ROOT / "results" / "eval"
    if src.resolve() == dst.resolve() or not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*_val_*"):
        shutil.copy2(f, dst / f.name)
        print(f"copied {f.name} -> {dst}")


def build_commands(a, out: Path, merged: Path) -> dict[str, list[str]]:
    py = sys.executable
    q = a.quick
    cmds = {
        "merge": [py, "scripts/merge_adapter.py", "--save"] + (["--adapter", a.adapter] if a.adapter else [])
        + (["--verify-prompts", "2"] if q else []),
        # split defaults to val; the test split is never requested
        "eval_merged_val": [py, "scripts/evaluate.py", "--merged", "--merged-dir", str(merged), "--tag", MERGED_TAG],
        "compute_metrics": [py, "scripts/compute_metrics.py", "--tags", "base", "finetuned", MERGED_TAG],
        "bench_kv_cache": [py, "scripts/bench_kv_cache.py", "--merged-dir", str(merged), "--mode", "both"]
        + (["--quick"] if q else []),
        "bench_attention": [py, "scripts/bench_attention.py"] + (["--quick"] if q else []),
    }
    return cmds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="pass --quick / fewer repeats to every benchmark and the merge check (val eval always covers the full val split)")
    ap.add_argument("--stop-on-failure", action="store_true", help="stop at the first failed step, benchmarks included")
    ap.add_argument("--keep-going", action="store_true", help="continue even if a setup step fails")
    ap.add_argument("--adapter", help="adapter dir or Hub id for the merge step (default: configs/inference.yaml)")
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS)
    ap.add_argument("--only", nargs="*", default=[], choices=STEPS)
    a = ap.parse_args()

    out = output_root(load_train_config())
    merged = default_merged_dir()
    logs = out / "logs"
    bench_dir = out / "results" / "benchmarks"
    logs.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)
    summary_path = bench_dir / "run_summary.json"
    cmds = build_commands(a, out, merged)
    summary = {"started": datetime.datetime.now().isoformat(timespec="seconds"), "quick": a.quick,
               "trialmatch_out": str(out), "merged_dir": str(merged), "steps": []}

    def save():
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    stopped = False
    for name in STEPS:
        if (a.only and name not in a.only) or name in a.skip:
            continue
        banner(name)
        rec = {"step": name, "log": str(logs / f"{name}.log")}
        t0 = time.time()
        if stopped:
            rec.update(status="not_run", note="earlier step failed")
        elif name == "bench_speculative":
            rec.update(status="skipped", note="placeholder: speculative decoding benchmark not implemented yet")
            print(rec["note"])
        elif name == "merge" and merged_dir_ready(merged):
            rec.update(status="skipped", note=f"merged model already exists: {merged}")
            print(rec["note"])
        else:
            if name == "env":
                code = step_env(logs / "env.log", out)
            else:
                if name == "compute_metrics":
                    sync_val_predictions(out)
                code = run_cmd(cmds[name], logs / f"{name}.log")
            rec["exit_code"] = code
            if code == 0:
                rec["status"] = "ok"
            elif name in BENCHMARKS and code == 1:
                rec.update(status="mismatch", note="benchmark reported a mismatch (exit 1); continuing")
            else:
                rec["status"] = "failed"
                if a.stop_on_failure or (name not in BENCHMARKS and not a.keep_going):
                    stopped = True
        rec["seconds"] = round(time.time() - t0, 1)
        summary["steps"].append(rec)
        summary["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        save()
        print(f"[{name}] -> {rec['status']} ({rec['seconds']} s)", flush=True)

    banner("SUMMARY")
    for r in summary["steps"]:
        print(f"{r['step']:<20}{r['status']:<10}{r.get('note', '')}")
    print(f"\nsummary: {summary_path}\nlogs:    {logs}")
    return 0 if all(r["status"] in ("ok", "skipped", "mismatch") for r in summary["steps"]) else 1


if __name__ == "__main__":
    sys.exit(main())
