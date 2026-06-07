from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()

import pandas as pd
import torch

from src.features.collagen_score import collagen_structure_features
from src.models.generator import Generator
from src.utils.metrics import ensure_dir, load_yaml
from src.utils.sequence_utils import AMINO_ACIDS
from src.utils.variable_length import min_max_lengths


def load_generator(config: dict, checkpoint: str | Path, device: torch.device) -> Generator:
    min_len, max_len = min_max_lengths(config)
    model = Generator(
        int(config["model"]["latent_dim"]),
        max_len,
        len(AMINO_ACIDS),
        int(config["model"]["hidden_dim"]),
        min_len=min_len,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def decode_probabilities(
    probs: torch.Tensor,
    config: dict,
    length_probs: torch.Tensor | None = None,
) -> list[str]:
    generation_cfg = config.get("generation", {})
    guided = bool(generation_cfg.get("collagen_guided_decoding", False))
    proline_boost = float(generation_cfg.get("xy_proline_boost", 1.0))
    adjusted = probs.clone()
    gly_idx = AMINO_ACIDS.index("G")
    pro_idx = AMINO_ACIDS.index("P")
    seq_len = adjusted.size(1)

    if guided:
        gly_prob = adjusted[:, :, gly_idx]
        frame_scores = []
        for frame in range(3):
            positions = list(range(frame, seq_len, 3))
            frame_scores.append(gly_prob[:, positions].mean(dim=1))
        best_frames = torch.stack(frame_scores, dim=1).argmax(dim=1).tolist()
        for row_index, frame in enumerate(best_frames):
            gly_positions = list(range(frame, seq_len, 3))
            adjusted[row_index, gly_positions, :] = 0.0
            adjusted[row_index, gly_positions, gly_idx] = 1.0
            if proline_boost != 1.0:
                xy_positions: list[int] = []
                for pos in range(frame, max(frame, seq_len - 2), 3):
                    if pos + 1 < seq_len:
                        xy_positions.append(pos + 1)
                    if pos + 2 < seq_len:
                        xy_positions.append(pos + 2)
                adjusted[row_index, xy_positions, pro_idx] *= proline_boost
                adjusted[row_index, xy_positions, :] /= adjusted[row_index, xy_positions, :].sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)

    sampled = torch.distributions.Categorical(probs=adjusted).sample().cpu().numpy()
    min_len, max_len = min_max_lengths(config)
    if length_probs is None:
        lengths = [max_len] * len(sampled)
    else:
        length_offsets = torch.distributions.Categorical(probs=length_probs).sample().cpu().numpy()
        lengths = [min_len + int(offset) for offset in length_offsets]
    return ["".join(AMINO_ACIDS[int(i)] for i in row[:length]) for row, length in zip(sampled, lengths)]


def generate_peptides(config: dict, checkpoint: str | Path | None = None) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = checkpoint or Path(config["paths"]["checkpoint_dir"]) / "generator.pt"
    generator = load_generator(config, checkpoint, device)
    n = int(config["generation"]["num_samples"])
    latent_dim = int(config["model"]["latent_dim"])
    temperature = float(config["generation"]["temperature"])
    batch_size = min(int(config["generation"].get("batch_size", 1024)), n)
    max_rounds = int(config["generation"].get("max_rounds", 30))
    sequences: list[str] = []
    seen: set[str] = set()
    with torch.no_grad():
        rounds = 0
        while len(sequences) < n and rounds < max_rounds:
            rounds += 1
            current_batch = max(batch_size, min(batch_size * 2, (n - len(sequences)) * 2))
            z = torch.randn(current_batch, latent_dim, device=device)
            probs, length_probs = generator(
                z,
                temperature=temperature,
                length_temperature=float(config["generation"].get("length_temperature", temperature)),
                return_length=True,
            )
            for sequence in decode_probabilities(probs, config, length_probs):
                if sequence in seen:
                    continue
                seen.add(sequence)
                sequences.append(sequence)
                if len(sequences) >= n:
                    break
    structure_rows = [collagen_structure_features(sequence).to_dict() for sequence in sequences]
    df = pd.DataFrame(structure_rows)
    output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "generated")
    output = output_dir / "generated_peptides.csv"
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} generated peptides to {output}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    generate_peptides(load_yaml(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
