from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()

import numpy as np
import pandas as pd

from src.utils.metrics import load_yaml
from src.utils.sequence_utils import one_hot_encode


def encode_csv(input_csv: str | Path, output_npz: str | Path, seq_len: int) -> None:
    df = pd.read_csv(input_csv)
    arrays = np.stack([one_hot_encode(seq, seq_len) for seq in df["sequence"]]).astype(np.float32)
    labels = df["label"].to_numpy(dtype=np.float32) if "label" in df.columns else np.zeros(len(df), dtype=np.float32)
    Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, x=arrays, y=labels, sequence=df["sequence"].astype(str).to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    input_csv = args.input or cfg["paths"]["train_dataset_csv"]
    output_npz = args.output or "data/processed/train_dataset_encoded.npz"
    encode_csv(input_csv, output_npz, int(cfg["data"]["seq_len"]))
    print(f"Saved encoded dataset to {output_npz}")


if __name__ == "__main__":
    main()
