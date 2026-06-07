from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        vocab_size: int,
        hidden_dim: int,
        min_len: int | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.min_len = min_len or seq_len
        self.length_classes = seq_len - self.min_len + 1
        if self.length_classes < 1:
            raise ValueError("min_len must be less than or equal to seq_len.")
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
        )
        self.sequence_head = nn.Linear(hidden_dim * 2, seq_len * vocab_size)
        self.length_head = nn.Linear(hidden_dim * 2, self.length_classes)

    def forward(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        length_temperature: float | None = None,
        return_length: bool = False,
    ):
        hidden = self.backbone(z)
        logits = self.sequence_head(hidden).view(-1, self.seq_len, self.vocab_size)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        length_logits = self.length_head(hidden)
        length_probs = torch.softmax(length_logits / (length_temperature or temperature), dim=-1)
        if return_length:
            return probabilities, length_probs
        return probabilities
