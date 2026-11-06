"""Dataset download/loading for the TrialGPT Criterion Annotations data."""
from pathlib import Path

import pandas as pd

from trialmatch.utils.common import load_config, resolve


def download_raw(cfg: dict | None = None, force: bool = False) -> Path:
    """Download the HF dataset to the raw parquet path. Idempotent unless force."""
    cfg = cfg or load_config()
    out = resolve(cfg["paths"]["raw"])
    if out.exists() and not force:
        print(f"[skip] raw file exists: {out} (use --force to re-download)")
        return out
    from datasets import load_dataset  # imported lazily; only needed for download

    ds = load_dataset(cfg["dataset"]["hf_id"], split=cfg["dataset"]["hf_split"])
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out))
    print(f"[ok] saved {len(ds)} rows to {out}")
    return out


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve(cfg["paths"]["raw"])
    if not path.exists():
        raise FileNotFoundError(f"Raw data missing: {path}. Run scripts/prepare_data.py first.")
    return pd.read_parquet(path)
