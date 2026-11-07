"""CLI: benchmark reference vs tiled online-softmax attention vs torch SDPA backends (comparison only).

  python scripts/bench_attention.py                 # GPU: fp16, seq 128..4096, blocks 32/64/128/256, 10 repeats
  python scripts/bench_attention.py --quick         # small sweep, 2 repeats (also fine on CPU, fp32)
Writes results/benchmarks/attention_<timestamp>.{json,csv,png} (under $TRIALMATCH_OUT if set).
Exit code 1 if a tiled/reference result differs from the fp32 baseline by more than the tolerance. No model, no data.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto", help="auto: fp16 on GPU, fp32 on CPU")
    ap.add_argument("--quick", action="store_true", help="seq 128/256/512, blocks 64/128, 2 repeats, 1 warm-up")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    if a.quick:
        a.seq_lens, a.block_sizes, a.repeats, a.warmup = [128, 256, 512], [64, 128], 2, 1

    import torch

    from trialmatch.inference.benchmarks import attention_bench as ab
    from trialmatch.training.config import load_train_config, output_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"auto": torch.float16 if device == "cuda" else torch.float32,
             "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    env = ab.environment(device, dtype)
    print(env)
    results = ab.run(a.seq_lens, a.block_sizes, a.heads, a.kv_heads, a.head_dim, device, dtype, a.warmup, a.repeats)
    out_dir = Path(a.out_dir) if a.out_dir else output_root(load_train_config()) / "results" / "benchmarks"
    jp, cp, pp = ab.write_results(out_dir, env, vars(a), results)
    bad = ab.mismatches(results)
    print(f"\nconfigs outside tolerance vs fp32 baseline: {len(bad)}\nsaved: {jp}\n       {cp}\n       {pp}")
    for r in bad:
        print(f"  MISMATCH {r['impl']} seq={r['seq_len']} max_err={r['max_abs_err']:.3e} > {r['tolerance']}")
    sys.exit(1 if bad else 0)
