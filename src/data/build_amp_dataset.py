from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()

import pandas as pd

from src.utils.metrics import load_yaml
from src.utils.sequence_utils import clean_sequence, is_valid_peptide


def load_amp_sources(raw_dir: str | Path, min_len: int, max_len: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(raw_dir).glob("*.csv")):
        if path.name.startswith("non_amp"):
            continue
        df = pd.read_csv(path)
        if "sequence" not in df.columns:
            continue
        df["sequence"] = df["sequence"].map(clean_sequence)
        df["source_file"] = path.name
        df["label"] = 1
        df = df[df["sequence"].map(lambda seq: is_valid_peptide(seq, min_len, max_len))]
        frames.append(df[["sequence", "source_file", "label"]])
    if not frames:
        raise ValueError(f"No AMP CSV files with a sequence column found in {raw_dir}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("sequence")


def build_amp_dataset(config: dict) -> pd.DataFrame:
    paths = config["paths"]
    data_cfg = config["data"]
    output = Path(paths["amp_positive_csv"])
    try:
        positives = load_amp_sources(paths["raw_dir"], data_cfg["min_len"], data_cfg["max_len"])
    except ValueError:
        if not output.exists():
            raise
        positives = pd.read_csv(output)
        if "sequence" not in positives.columns:
            raise ValueError(f"Existing AMP file has no sequence column: {output}")
        positives["sequence"] = positives["sequence"].map(clean_sequence)
        positives = positives[
            positives["sequence"].map(lambda seq: is_valid_peptide(seq, data_cfg["min_len"], data_cfg["max_len"]))
        ].copy()
        if "source_file" not in positives.columns:
            positives["source_file"] = output.name
        positives["label"] = 1
        positives = positives[["sequence", "source_file", "label"]].drop_duplicates("sequence")
    output.parent.mkdir(parents=True, exist_ok=True)
    positives.to_csv(output, index=False)
    return positives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    args = parser.parse_args()
    df = build_amp_dataset(load_yaml(args.config))
    print(f"Saved {len(df)} AMP positives.")


if __name__ == "__main__":
    main()
