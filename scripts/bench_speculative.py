"""CLI: benchmark greedy speculative decoding against plain cached greedy decoding (batch size 1).

  python scripts/bench_speculative.py --merged-dir $TRIALMATCH_OUT/merged/qwen3-4b_merged_fp16
  python scripts/bench_speculative.py --merged-dir ... --resume            # continue the latest speculative_*.json
  python scripts/bench_speculative.py --merged-dir ... --quick --max-configs 6
  python scripts/bench_speculative.py --tiny-random --quick                # CPU smoke run, no downloads

Target = merged fp16 Qwen3-4B (same loader as the KV-cache benchmark). Draft = Qwen/Qwen3-0.6B (untuned) by default;
--draft-model / --draft-dir / --draft-adapter select another (e.g. a fine-tuned draft trained later).
Writes results/benchmarks/speculative_<timestamp>.{json,csv} after EVERY configuration, and the plot at the end.
Only the val split is used for real prompts; the test split is never read.
Exit code 1 if any speculative output differs from plain greedy decoding (fp16 near-ties are reported, not hidden).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-dir", help="saved merged fp16 target directory")
    ap.add_argument("--adapter", help="target adapter dir/Hub id: merge on the fly instead of loading a saved model")
    ap.add_argument("--tiny-random", action="store_true", help="tiny random target+draft (CPU smoke test only)")
    ap.add_argument("--draft-model", help="draft Hub id (default from configs/inference.yaml: Qwen/Qwen3-0.6B)")
    ap.add_argument("--draft-dir", help="local dir of a fully fine-tuned draft (overrides --draft-model)")
    ap.add_argument("--draft-adapter", help="LoRA adapter merged into the draft")
    ap.add_argument("--mode", choices=["synthetic", "val", "both"], default="both")
    ap.add_argument("--k", type=int, nargs="+", help="k sweep (default 1 2 4 8)")
    ap.add_argument("--input-lengths", type=int, nargs="+")
    ap.add_argument("--output-lengths", type=int, nargs="+")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--val-prompts", type=int)
    ap.add_argument("--quick", action="store_true", help="inputs 128/256, outputs 16/64, k 2 4, 2 repeats, 2 val prompts")
    ap.add_argument("--max-configs", type=int, help="stop after this many NEW configurations (resume later)")
    ap.add_argument("--resume", action="store_true", help="continue the latest speculative_*.json, skipping finished configurations")
    ap.add_argument("--out-dir")
    a = ap.parse_args()

    import torch

    from trialmatch.inference.benchmarks import kv_cache_bench as kb
    from trialmatch.inference.benchmarks import speculative_bench as sb
    from trialmatch.model.load import inference_config, pick_device_dtype
    from trialmatch.training.config import load_train_config, output_root

    bc = inference_config()["speculative_benchmark"]
    ins = a.input_lengths or ([128, 256] if a.quick else bc["input_lengths"])
    outs = a.output_lengths or ([16, 64] if a.quick else bc["output_lengths"])
    ks = a.k or ([2, 4] if a.quick else bc["k_sweep"])
    reps = a.repeats or (2 if a.quick else bc["repeats"])
    warm = bc["warmup"] if a.warmup is None else a.warmup
    nval = a.val_prompts or (2 if a.quick else bc["val_prompts"])
    torch.manual_seed(bc["seed"])

    tok = None
    if a.tiny_random:
        from trialmatch.inference.benchmarks.tiny_model import build_tiny_draft, build_tiny_qwen3
        device, dtype = "cpu", torch.float32
        mp = max(ins) + max(outs) + 16
        target, draft = build_tiny_qwen3(bc["seed"], max_pos=mp), build_tiny_draft(max_pos=mp)
        draft_label = "tiny-random-draft"
        if a.mode != "synthetic":
            sys.exit("--tiny-random has no tokenizer: use --mode synthetic")
    else:
        from trialmatch.inference.speculative.loading import check_compatible, load_draft
        from trialmatch.model.load import load_merged_for_inference
        device, dtype = pick_device_dtype()
        target, tok = load_merged_for_inference(a.adapter, a.merged_dir)
        dm = a.draft_model or bc["draft_model"]
        dd, da = a.draft_dir or bc.get("draft_dir"), a.draft_adapter or bc.get("draft_adapter")
        draft = load_draft(dm, dd, da, device, dtype)
        check_compatible(target, draft)
        draft_label = dm + (f" + adapter {da}" if da else "") + (f" [dir {dd}]" if dd else "")
    target.eval()
    draft.eval()
    env = sb.environment(target, draft, device, next(target.parameters()).dtype, draft_label)
    print(env, flush=True)

    val_cases, eos_val = [], None
    base_ids = None
    if tok is not None:
        from trialmatch.model.chat import end_token_id
        eos_val = end_token_id(tok)
        base_ids = kb.val_inputs(tok, 1)[0][1][0].tolist()
        if a.mode in ("val", "both"):
            val_cases = kb.val_inputs(tok, nval)
    cfgs = sb.plan(ins if a.mode in ("synthetic", "both") else [], outs, ks, draft_label, val_cases,
                   bc["val_max_new_tokens"])
    val_ids = {aid: ids for aid, ids in val_cases}

    def inputs_for(c):
        return val_ids[c["aid"]] if c["source"] == "val" else kb.synthetic_input(c["input_len"], target.config.vocab_size, bc["seed"], base_ids)

    def eos_for(c):
        return eos_val if c["source"] == "val" else None

    out_dir = Path(a.out_dir) if a.out_dir else output_root(load_train_config()) / "results" / "benchmarks"
    args = {**vars(a), "input_lengths": ins, "output_lengths": outs, "k": ks, "repeats": reps, "warmup": warm,
            "val_prompts": nval, "batch_size": 1}
    print(f"{len(cfgs)} configurations planned", flush=True)
    rows = sb.run_benchmark(target, draft, cfgs, inputs_for, eos_for, out_dir, env, args, device, warm, reps,
                            resume=a.resume, max_configs=a.max_configs)
    stamp = args["stamp"]
    pp = out_dir / f"speculative_{stamp}.png"
    sb.plot(rows, pp, env)
    bad = [r for r in rows if not r["tokens_equal"]]
    print(f"\nrows: {len(rows)}; speculative == plain greedy tokens in all rows: {not bad}")
    for r in bad:
        print("MISMATCH", r["key"], r.get("mismatch_report"))
    print(f"saved: {out_dir / f'speculative_{stamp}.json'}\n       {out_dir / f'speculative_{stamp}.csv'}\n       {pp}")
    sys.exit(1 if bad else 0)
