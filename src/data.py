"""Data loading utilities — California (runnable) + Kaggle helper for Ames.

Usage examples:
- python -m src.data --write-sample   # writes a sample CSV to data/raw/
- from src.data import load_california
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing

ROOT = Path(__file__).resolve().parents[1]


def load_california(save_csv: str | None = None, sample_n: int | None = None) -> pd.DataFrame:
    """Load California housing into a DataFrame. Optionally save a small CSV sample.

    Returns DataFrame with features and target column `MedHouseVal`.
    """
    bunch = fetch_california_housing(as_frame=True)
    df = pd.concat([bunch.data, bunch.target.rename("MedHouseVal")], axis=1)
    if sample_n:
        df = df.sample(min(sample_n, len(df)), random_state=42)
    if save_csv:
        out = Path(save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def fetch_ames_kaggle(output_dir: str | None = None) -> str:
    """Download Ames dataset via kaggle API if available.

    - Requires Kaggle API token at %USERPROFILE%\\.kaggle\\kaggle.json
    - If API is not configured, raises RuntimeError with instructions.
    """
    try:
        import kaggle
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "kaggle package not available or API not configured. "
            "Follow README to add %USERPROFILE%\\.kaggle\\kaggle.json or install the kaggle package."
        ) from exc
    out = Path(output_dir or ROOT / "data" / "raw")
    out.mkdir(parents=True, exist_ok=True)
    # competition name for Ames
    kaggle.api.competition_download_files("house-prices-advanced-regression-techniques", path=str(out))
    return str(out)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--write-sample", action="store_true", help="write a small California sample to data/raw/")
    p.add_argument("--sample-n", type=int, default=1000)
    args = p.parse_args()

    if args.write_sample:
        out = ROOT / "data" / "raw" / "california_sample_from_loader.csv"
        df = load_california(save_csv=str(out), sample_n=args.sample_n)
        print(f"Wrote sample CSV to: {out} (rows={len(df)})")
