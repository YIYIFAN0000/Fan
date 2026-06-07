from __future__ import annotations

from collections import Counter

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
INDEX_TO_AA = {i: aa for aa, i in AA_TO_INDEX.items()}


def clean_sequence(sequence: str) -> str:
    return "".join(aa for aa in str(sequence).upper() if aa in AA_TO_INDEX)


def is_valid_peptide(sequence: str, min_len: int = 5, max_len: int = 60) -> bool:
    cleaned = clean_sequence(sequence)
    return min_len <= len(cleaned) <= max_len and len(cleaned) == len(str(sequence).strip())


def pad_or_trim(sequence: str, seq_len: int, pad_token: str = "G") -> str:
    sequence = clean_sequence(sequence)
    if len(sequence) >= seq_len:
        return sequence[:seq_len]
    return sequence + pad_token * (seq_len - len(sequence))


def one_hot_encode(sequence: str, seq_len: int | None = None) -> np.ndarray:
    if seq_len is not None:
        sequence = pad_or_trim(sequence, seq_len)
    encoded = np.zeros((len(sequence), len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        encoded[i, AA_TO_INDEX[aa]] = 1.0
    return encoded


def one_hot_decode(array: np.ndarray) -> str:
    indices = np.argmax(array, axis=-1)
    return "".join(INDEX_TO_AA[int(index)] for index in indices)


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def repeat_fraction(sequence: str) -> float:
    sequence = clean_sequence(sequence)
    if not sequence:
        return 0.0
    return max(Counter(sequence).values()) / len(sequence)
