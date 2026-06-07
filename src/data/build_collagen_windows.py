from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()

import pandas as pd

from src.features.collagen_score import collagen_structure_features
from src.utils.fasta_utils import read_fasta
from src.utils.metrics import load_yaml
from src.utils.variable_length import length_range


def make_collagen_windows(
    fasta_path: str | Path,
    seq_lengths: list[int],
    stride: int,
    min_unique_ratio: float,
    min_collagen_motif_score: float = 0.0,
    min_glycine_frame_fraction: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for record_id, sequence in read_fasta(fasta_path):
        for seq_len in seq_lengths:
            for start in range(0, max(0, len(sequence) - seq_len + 1), stride):
                window = sequence[start : start + seq_len]
                if len(window) != seq_len:
                    continue
                if len(set(window)) / seq_len < min_unique_ratio:
                    continue
                structure = collagen_structure_features(window).to_dict()
                if structure["collagen_motif_score"] < min_collagen_motif_score:
                    continue
                if structure["glycine_frame_fraction"] < min_glycine_frame_fraction:
                    continue
                structure.update({"record_id": record_id, "start": start, "label": 0})
                rows.append(structure)
    if not rows:
        raise ValueError("No collagen windows created. Check FASTA path and config.")
    return pd.DataFrame(rows).drop_duplicates("sequence")


def build_collagen_windows(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    paths = config["paths"]
    min_len = int(data_cfg.get("collagen_window_min_len", data_cfg.get("min_len", data_cfg["seq_len"])))
    max_len = int(data_cfg.get("collagen_window_max_len", data_cfg.get("max_len", data_cfg["seq_len"])))
    fasta_path = Path(paths["sheep_collagen_fasta"])
    output = Path(paths["collagen_windows_csv"])
    if fasta_path.exists():
        df = make_collagen_windows(
            fasta_path,
            length_range(min_len, max_len),
            int(data_cfg["collagen_stride"]),
            float(data_cfg["min_unique_ratio"]),
            float(data_cfg.get("min_collagen_motif_score", 0.0)),
            float(data_cfg.get("min_glycine_frame_fraction", 0.0)),
        )
    elif output.exists():
        df = pd.read_csv(output)
        df = df[(df["length"] >= min_len) & (df["length"] <= max_len)].copy()
        if "collagen_motif_score" in df.columns:
            df = df[df["collagen_motif_score"] >= float(data_cfg.get("min_collagen_motif_score", 0.0))]
        if "glycine_frame_fraction" in df.columns:
            df = df[df["glycine_frame_fraction"] >= float(data_cfg.get("min_glycine_frame_fraction", 0.0))]
        if "label" not in df.columns:
            df["label"] = 0
        if df.empty:
            raise ValueError(f"Existing collagen window file has no rows after filtering: {output}")
    else:
        raise FileNotFoundError(
            f"Missing collagen FASTA {fasta_path} and no processed fallback exists at {output}."
        )
    max_windows = int(data_cfg.get("max_collagen_windows", 0))
    if max_windows > 0 and len(df) > max_windows:
        df = (
            df.sort_values(["collagen_motif_score", "glycine_frame_fraction"], ascending=[False, False])
            .head(max_windows)
            .sort_values(["record_id", "start", "length"])
            .reset_index(drop=True)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    args = parser.parse_args()
    df = build_collagen_windows(load_yaml(args.config))
    print(f"Saved {len(df)} collagen windows.")


if __name__ == "__main__":
    main()
