from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()

import pandas as pd

from src.utils.sequence_utils import clean_sequence, is_valid_peptide


def clean_csv(input_path: str | Path, output_path: str | Path, min_len: int, max_len: int) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if "sequence" not in df.columns:
        raise ValueError(f"{input_path} must contain a sequence column.")
    df["sequence"] = df["sequence"].map(clean_sequence)
    df = df[df["sequence"].map(lambda seq: is_valid_peptide(seq, min_len, max_len))].copy()
    df = df.drop_duplicates("sequence")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=50)
    args = parser.parse_args()
    df = clean_csv(args.input, args.output, args.min_len, args.max_len)
    print(f"Saved {len(df)} cleaned sequences to {args.output}")


if __name__ == "__main__":
    main()
