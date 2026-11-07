"""CLI: benchmark the manual KV-cache decoder against the uncached reference (same model, same inputs).

  python scripts/bench_kv_cache.py --merged-dir $TRIALMATCH_OUT/merged/qwen3-4b_merged_fp16 --mode both
  python scripts/bench_kv_cache.py --tiny-random --quick        # CPU smoke run, no downloads
Writes results/benchmarks/kv_cache_<timestamp>.{json,csv,png} (under $TRIALMATCH_OUT if set).
Only the val split is used for real prompts; the test split is never read.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-dir", help="saved merged fp16 model directory")
    ap.add_argument("--adapter", help="adapter dir/Hub id: merge on the fly instead of loading a saved model")
    ap.add_argument("--tiny-random", action="store_true", help="tiny random Qwen3 (CPU smoke test only)")
    ap.add_argument("--mode", choices=["synthetic", "val", "both"], default="both")
    ap.add_argument("--input-lengths", type=int, nargs="+")
    ap.add_argument("--output-lengths", type=int, nargs="+")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--val-prompts", type=int)
    ap.add_argument("--quick", action="store_true", help="inputs 128/256, outputs 16/64, 2 repeats, 4 val prompts")
    ap.add_argument("--out-dir")
    a = ap.parse_args()

    import torch

    from trialmatch.inference.benchmarks import kv_cache_bench as kb
    from trialmatch.model.load import inference_config, pick_device_dtype
    from trialmatch.training.config import load_train_config, output_root

    bc = inference_config()["kv_cache_benchmark"]
    ins = a.input_lengths or ([128, 256] if a.quick else bc["input_lengths"])
    outs = a.output_lengths or ([16, 64] if a.quick else bc["output_lengths"])
    reps = a.repeats or (2 if a.quick else bc["repeats"])
    warm = bc["warmup"] if a.warmup is None else a.warmup
    nval = a.val_prompts or (4 if a.quick else bc["val_prompts"])
    torch.manual_seed(bc["seed"])

    tok = None
    if a.tiny_random:
        from trialmatch.inference.benchmarks.tiny_model import build_tiny_qwen3
        device, dtype = "cpu", torch.float32
        model = build_tiny_qwen3(bc["seed"], max_pos=max(ins) + max(outs) + 8)
        if a.mode != "synthetic":
            sys.exit("--tiny-random has no tokenizer: use --mode synthetic")
    else:
        from trialmatch.model.load import load_merged_for_inference
        device, dtype = pick_device_dtype()
        model, tok = load_merged_for_inference(a.adapter, a.merged_dir)
    model.eval()
    vocab = model.config.vocab_size
    env = kb.environment(model, device, next(model.parameters()).dtype)
    print(env)

    cases = []
    if a.mode in ("synthetic", "both"):
        base_ids = None
        if tok is not None:  # tile a real val prompt so synthetic text is natural; falls back to random ids
            base_ids = kb.val_inputs(tok, 1)[0][1][0].tolist()
        cases += kb.run_grid(model, device, vocab, ins, outs, warm, reps, bc["seed"], base_ids)
    if a.mode in ("val", "both"):
        cases += kb.run_val(model, tok, device, nval, bc["val_max_new_tokens"], warm, reps)

    out_dir = Path(a.out_dir) if a.out_dir else output_root(load_train_config()) / "results" / "benchmarks"
    args = {**vars(a), "input_lengths": ins, "output_lengths": outs, "repeats": reps, "warmup": warm,
            "val_prompts": nval}
    jp, cp, pp = kb.write_results(out_dir, env, args, cases)
    ok = all(c["tokens_equal"] for c in cases)
    print(f"\nall cached==uncached token sequences equal: {ok}\nsaved: {jp}\n       {cp}\n       {pp}")
    sys.exit(0 if ok else 1)
