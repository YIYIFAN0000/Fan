from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .peptide_utils import AA_TO_INDEX, AMINO_ACIDS, clean_sequence


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_fasta(path: str | Path) -> list[str]:
    fasta_path = Path(path)
    if not fasta_path.is_absolute():
        fasta_path = project_root() / fasta_path
    records: list[str] = []
    current: list[str] = []
    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    records.append(clean_sequence("".join(current)))
                    current = []
            else:
                current.append(line)
        if current:
            records.append(clean_sequence("".join(current)))
    return [record for record in records if record]


def make_windows(sequences: list[str], seq_len: int, stride: int, min_unique_ratio: float) -> list[str]:
    windows: list[str] = []
    for sequence in sequences:
        if len(sequence) < seq_len:
            continue
        for start in range(0, len(sequence) - seq_len + 1, stride):
            window = sequence[start : start + seq_len]
            if len(set(window)) / seq_len >= min_unique_ratio:
                windows.append(window)
    if not windows:
        raise ValueError("No peptide windows were created. Check FASTA length and filtering settings.")
    return windows


def one_hot_encode(sequence: str) -> np.ndarray:
    encoded = np.zeros((len(sequence), len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        encoded[i, AA_TO_INDEX[aa]] = 1.0
    return encoded


def one_hot_decode(array: np.ndarray) -> str:
    indices = np.argmax(array, axis=-1)
    return "".join(AMINO_ACIDS[int(index)] for index in indices)


def load_training_array(config: dict) -> tuple[np.ndarray, list[str]]:
    data_config = config["data"]
    records = read_fasta(data_config["fasta_path"])
    windows = make_windows(
        records,
        seq_len=int(data_config["seq_len"]),
        stride=int(data_config["stride"]),
        min_unique_ratio=float(data_config["min_unique_ratio"]),
    )
    array = np.stack([one_hot_encode(window) for window in windows]).astype(np.float32)
    return array, windows


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = project_root() / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
