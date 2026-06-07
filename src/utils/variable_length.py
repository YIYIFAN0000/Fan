from __future__ import annotations

import numpy as np
import torch

from src.utils.sequence_utils import AMINO_ACIDS, AA_TO_INDEX, clean_sequence


def min_max_lengths(config: dict) -> tuple[int, int]:
    data_cfg = config["data"]
    min_len = int(data_cfg.get("min_len", data_cfg.get("seq_len", 1)))
    max_len = int(data_cfg.get("max_len", data_cfg["seq_len"]))
    return min_len, max_len


def length_range(min_len: int, max_len: int) -> list[int]:
    return list(range(int(min_len), int(max_len) + 1))


def one_hot_encode_with_mask(sequence: str, max_len: int) -> np.ndarray:
    cleaned = clean_sequence(sequence)[:max_len]
    encoded = np.zeros((max_len, len(AMINO_ACIDS) + 1), dtype=np.float32)
    for index, aa in enumerate(cleaned):
        encoded[index, AA_TO_INDEX[aa]] = 1.0
        encoded[index, -1] = 1.0
    return encoded


def soft_length_mask(length_probs: torch.Tensor, min_len: int, max_len: int) -> torch.Tensor:
    device = length_probs.device
    positions = torch.arange(1, max_len + 1, device=device).view(1, -1)
    lengths = torch.arange(min_len, max_len + 1, device=device).view(-1, 1)
    support = (positions <= lengths).float()
    return length_probs @ support


def append_mask_channel(probabilities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_probabilities = probabilities * mask.unsqueeze(-1)
    return torch.cat([masked_probabilities, mask.unsqueeze(-1)], dim=-1)


def target_length_distribution(sequences: list[str], min_len: int, max_len: int, device: torch.device) -> torch.Tensor:
    counts = torch.ones(max_len - min_len + 1, device=device) * 1e-3
    for sequence in sequences:
        length = len(clean_sequence(sequence))
        if min_len <= length <= max_len:
            counts[length - min_len] += 1.0
    return counts / counts.sum()


def length_distribution_loss(length_probs: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    generated = length_probs.mean(dim=0).clamp_min(1e-8)
    target = target_distribution.clamp_min(1e-8)
    return torch.sum(target * (torch.log(target) - torch.log(generated)))
