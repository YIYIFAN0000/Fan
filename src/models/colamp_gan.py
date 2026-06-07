from __future__ import annotations

from dataclasses import dataclass

from src.models.critic import Critic
from src.models.generator import Generator


@dataclass
class ColAMPGAN:
    generator: Generator
    critic: Critic
