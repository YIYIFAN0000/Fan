from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.features.esm2_embeddings import (
    esm_feature_dim,
    load_esm_amino_acid_embeddings,
    soft_esm_features,
    use_esm_features,
)
from src.models.amp_predictor import ESMFeaturePredictor, SequencePredictor
from src.models.critic import Critic
from src.models.generator import Generator
from src.models.toxicity_predictor import ToxicityPredictor
from src.training.losses import gradient_penalty
from src.training.train_predictors import prepare_train_dataset, train_predictors
from src.utils.metrics import ensure_dir, load_yaml, set_seed
from src.utils.sequence_utils import AMINO_ACIDS
from src.utils.variable_length import (
    append_mask_channel,
    length_distribution_loss,
    min_max_lengths,
    one_hot_encode_with_mask,
    soft_length_mask,
    target_length_distribution,
)


def checkpoint_shape_matches(checkpoint: dict, config: dict) -> bool:
    state = checkpoint.get("model_state", {})
    hidden_dim = int(config["model"]["predictor_hidden_dim"])
    if use_esm_features(config):
        input_dim = esm_feature_dim(config)
        mid_dim = max(hidden_dim // 2, 8)
        return (
            tuple(state.get("net.1.weight", torch.empty(0)).shape) == (hidden_dim, input_dim)
            and tuple(state.get("net.4.weight", torch.empty(0)).shape) == (mid_dim, hidden_dim)
            and tuple(state.get("net.7.weight", torch.empty(0)).shape) == (1, mid_dim)
        )

    vocab_size = len(AMINO_ACIDS)
    mid_dim = max(hidden_dim // 2, 8)
    return (
        tuple(state.get("features.0.weight", torch.empty(0)).shape) == (hidden_dim, vocab_size, 3)
        and tuple(state.get("features.2.weight", torch.empty(0)).shape) == (hidden_dim, hidden_dim, 5)
        and tuple(state.get("classifier.1.weight", torch.empty(0)).shape) == (mid_dim, hidden_dim)
        and tuple(state.get("classifier.4.weight", torch.empty(0)).shape) == (1, mid_dim)
    )


def load_predictor(path: str | Path, model_cls, config: dict, device: torch.device):
    if use_esm_features(config):
        model = ESMFeaturePredictor(
            esm_feature_dim(config),
            int(config["model"]["predictor_hidden_dim"]),
        ).to(device)
    else:
        seq_len = int(config["data"]["seq_len"])
        model = model_cls(seq_len, len(AMINO_ACIDS), int(config["model"]["predictor_hidden_dim"])).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def predictors_ready(predictor_dir: Path, config: dict) -> bool:
    expected_esm = use_esm_features(config)
    expected_hidden_dim = int(config["model"]["predictor_hidden_dim"])
    expected_esm_cfg = config.get("esm", {})
    expected_data_cfg = config.get("data", {})
    for filename in ("amp_predictor.pt", "toxicity_predictor.pt"):
        path = predictor_dir / filename
        if not path.exists():
            return False
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config", {})
        saved_model_cfg = saved_config.get("model", {})
        if not checkpoint_shape_matches(checkpoint, config):
            return False
        if int(saved_model_cfg.get("predictor_hidden_dim", 0)) != expected_hidden_dim:
            return False
        saved_data_cfg = saved_config.get("data", {})
        for key in ("seq_len", "min_len", "max_len", "variable_length_version"):
            if saved_data_cfg.get(key) != expected_data_cfg.get(key):
                return False
        if saved_data_cfg.get("negative_strategy_version") != expected_data_cfg.get("negative_strategy_version"):
            return False
        if float(saved_data_cfg.get("min_collagen_motif_score", 0.0)) != float(
            expected_data_cfg.get("min_collagen_motif_score", 0.0)
        ):
            return False
        if float(saved_data_cfg.get("min_glycine_frame_fraction", 0.0)) != float(
            expected_data_cfg.get("min_glycine_frame_fraction", 0.0)
        ):
            return False
        if use_esm_features(saved_config) != expected_esm:
            return False
        if expected_esm:
            saved_esm_cfg = saved_config.get("esm", {})
            if int(saved_esm_cfg.get("embedding_dim", 0)) != esm_feature_dim(config):
                return False
            if saved_esm_cfg.get("model_name") != expected_esm_cfg.get("model_name"):
                return False
            if saved_esm_cfg.get("embedding_mode") != expected_esm_cfg.get("embedding_mode"):
                return False
    return True


def _mean_positions(values: torch.Tensor, positions: list[int], mask: torch.Tensor | None = None) -> torch.Tensor:
    if not positions:
        return torch.zeros(values.size(0), device=values.device, dtype=values.dtype)
    if mask is not None:
        weights = mask[:, positions]
        return (values[:, positions] * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    return values[:, positions].mean(dim=1)


def collagen_soft_reward(
    fake: torch.Tensor,
    mask: torch.Tensor | None = None,
    frame_sharpness: float = 8.0,
) -> torch.Tensor:
    gly_idx = AMINO_ACIDS.index("G")
    pro_idx = AMINO_ACIDS.index("P")
    gly_prob = fake[:, :, gly_idx]
    pro_prob = fake[:, :, pro_idx]
    seq_len = fake.size(1)

    frame_scores: list[torch.Tensor] = []
    for frame in range(3):
        gly_positions = list(range(frame, seq_len, 3))
        xy_positions: list[int] = []
        for pos in range(frame, max(frame, seq_len - 2), 3):
            if pos + 1 < seq_len:
                xy_positions.append(pos + 1)
            if pos + 2 < seq_len:
                xy_positions.append(pos + 2)
        gly_score = _mean_positions(gly_prob, gly_positions, mask)
        xy_pro_score = _mean_positions(pro_prob, xy_positions, mask)
        frame_scores.append(0.70 * gly_score + 0.30 * xy_pro_score)

    stacked = torch.stack(frame_scores, dim=1)
    frame_weights = torch.softmax(stacked * frame_sharpness, dim=1)
    return (stacked * frame_weights).sum(dim=1).mean()


def batch_diversity_reward(fake: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if fake.size(0) < 2:
        return torch.zeros((), device=fake.device, dtype=fake.dtype)
    if mask is not None:
        fake = fake * mask.unsqueeze(-1)
    flat = fake.reshape(fake.size(0), -1)
    return torch.pdist(flat, p=1).mean() / max(1, fake.size(1) * 2)


def _masked_fraction(fake: torch.Tensor, mask: torch.Tensor, amino_acids: str) -> torch.Tensor:
    indices = [AMINO_ACIDS.index(aa) for aa in amino_acids]
    numerator = fake[:, :, indices].sum(dim=-1)
    return (numerator * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)


def composition_penalty(fake: torch.Tensor, mask: torch.Tensor, config: dict) -> torch.Tensor:
    cfg = config["gan"]
    positive_fraction = _masked_fraction(fake, mask, "KRH")
    arginine_fraction = _masked_fraction(fake, mask, "R")
    lysine_fraction = _masked_fraction(fake, mask, "K")
    proline_fraction = _masked_fraction(fake, mask, "P")
    glycine_fraction = _masked_fraction(fake, mask, "G")
    charge_density = (
        _masked_fraction(fake, mask, "KRH")
        - _masked_fraction(fake, mask, "DE")
    )
    penalties = [
        torch.relu(positive_fraction - float(cfg.get("max_positive_fraction", 0.34))),
        torch.relu(arginine_fraction - float(cfg.get("max_arginine_fraction", 0.18))),
        torch.relu(lysine_fraction - float(cfg.get("max_lysine_fraction", 0.18))),
        torch.relu(proline_fraction - float(cfg.get("max_proline_fraction", 0.34))),
        torch.relu(glycine_fraction - float(cfg.get("max_glycine_fraction", 0.40))),
        torch.relu(charge_density - float(cfg.get("max_charge_density", 0.28))),
    ]
    return torch.stack(penalties, dim=1).sum(dim=1).mean()


def low_complexity_penalty(fake: torch.Tensor, mask: torch.Tensor, config: dict) -> torch.Tensor:
    cfg = config["gan"]
    aa_distribution = (fake * mask.unsqueeze(-1)).sum(dim=1)
    aa_distribution = aa_distribution / aa_distribution.sum(dim=1, keepdim=True).clamp_min(1e-6)
    entropy = -(aa_distribution.clamp_min(1e-8) * aa_distribution.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / np.log(len(AMINO_ACIDS))
    return torch.relu(float(cfg.get("min_sequence_entropy", 0.55)) - entropy).mean()


def repeated_motif_penalty(fake: torch.Tensor, mask: torch.Tensor, config: dict) -> torch.Tensor:
    cfg = config["gan"]
    motifs = cfg.get("penalized_motifs", ["GPP", "PPP", "PGP", "PPG", "RRR", "KKK", "RKR", "KRK"])
    if fake.size(1) < 3 or not motifs:
        return torch.zeros((), device=fake.device, dtype=fake.dtype)
    valid = mask[:, :-2] * mask[:, 1:-1] * mask[:, 2:]
    if valid.sum() <= 0:
        return torch.zeros((), device=fake.device, dtype=fake.dtype)
    motif_scores: list[torch.Tensor] = []
    for motif in motifs:
        if len(motif) != 3 or any(aa not in AMINO_ACIDS for aa in motif):
            continue
        i, j, k = [AMINO_ACIDS.index(aa) for aa in motif]
        motif_scores.append(fake[:, :-2, i] * fake[:, 1:-1, j] * fake[:, 2:, k])
    if not motif_scores:
        return torch.zeros((), device=fake.device, dtype=fake.dtype)
    stacked = torch.stack(motif_scores, dim=0).sum(dim=0)
    return (stacked * valid).sum(dim=1).div(valid.sum(dim=1).clamp_min(1e-6)).mean()


def train_gan(config: dict) -> None:
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor_dir = Path(config["paths"]["predictor_dir"])
    if not predictors_ready(predictor_dir, config):
        train_predictors(config)

    train_csv = Path(config["paths"]["train_dataset_csv"])
    if not train_csv.exists():
        prepare_train_dataset(config)
    df = pd.read_csv(train_csv)
    min_len, max_len = min_max_lengths(config)
    seq_len = max_len
    positives = df[df["label"] == 1]["sequence"].tolist()
    if not positives:
        raise ValueError("No positive AMP sequences available for GAN training.")
    collagen_targets: list[str] = []
    collagen_csv = Path(config["paths"]["collagen_windows_csv"])
    real_collagen_fraction = float(config["gan"].get("real_collagen_fraction", 0.0))
    if real_collagen_fraction > 0 and collagen_csv.exists():
        collagen_df = pd.read_csv(collagen_csv)
        if "collagen_motif_score" in collagen_df.columns:
            collagen_df = collagen_df[
                collagen_df["collagen_motif_score"] >= float(config["data"].get("min_collagen_motif_score", 0.0))
            ]
        if "glycine_frame_fraction" in collagen_df.columns:
            collagen_df = collagen_df[
                collagen_df["glycine_frame_fraction"] >= float(config["data"].get("min_glycine_frame_fraction", 0.0))
            ]
        collagen_df = collagen_df[~collagen_df["sequence"].isin(set(positives))]
        if not collagen_df.empty:
            target_n = int(len(positives) * real_collagen_fraction / max(1e-6, 1.0 - real_collagen_fraction))
            target_n = max(1, min(len(collagen_df), target_n))
            collagen_targets = collagen_df.sample(n=target_n, random_state=int(config["seed"]))["sequence"].tolist()
            print(f"Using GAN real sequences: amp={len(positives)} collagen_like={len(collagen_targets)}")

    real_sequences = [sequence for sequence in list(dict.fromkeys([*positives, *collagen_targets])) if min_len <= len(sequence) <= max_len]
    x = torch.tensor(np.stack([one_hot_encode_with_mask(seq, seq_len) for seq in real_sequences]), dtype=torch.float32)
    batch_size = min(int(config["gan"]["batch_size"]), len(x))
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True, drop_last=len(x) >= batch_size)

    latent_dim = int(config["model"]["latent_dim"])
    hidden_dim = int(config["model"]["hidden_dim"])
    generator = Generator(latent_dim, seq_len, len(AMINO_ACIDS), hidden_dim, min_len=min_len).to(device)
    critic = Critic(seq_len, len(AMINO_ACIDS) + 1, hidden_dim).to(device)
    amp_predictor = load_predictor(predictor_dir / "amp_predictor.pt", SequencePredictor, config, device)
    tox_predictor = load_predictor(predictor_dir / "toxicity_predictor.pt", ToxicityPredictor, config, device)
    aa_esm_embeddings = load_esm_amino_acid_embeddings(config, device) if use_esm_features(config) else None

    opt_g = torch.optim.Adam(generator.parameters(), lr=float(config["gan"]["learning_rate"]), betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=float(config["gan"]["learning_rate"]), betas=(0.5, 0.9))
    target_lengths = target_length_distribution(real_sequences, min_len, max_len, device)

    ckpt_dir = ensure_dir(config["paths"]["checkpoint_dir"])
    history_rows: list[dict] = []
    for epoch in range(1, int(config["gan"]["epochs"]) + 1):
        epoch_stats = {
            "generator_loss": [],
            "critic_loss": [],
            "amp_reward": [],
            "toxicity_penalty": [],
            "collagen_reward": [],
            "diversity_reward": [],
            "composition_penalty": [],
            "low_complexity_penalty": [],
            "repeated_motif_penalty": [],
        }
        for (real,) in loader:
            real = real.to(device)
            current_batch = real.size(0)
            for _ in range(int(config["gan"]["critic_steps"])):
                z = torch.randn(current_batch, latent_dim, device=device)
                fake, length_probs = generator(z, return_length=True)
                fake_mask = soft_length_mask(length_probs, min_len, seq_len)
                fake_critic = append_mask_channel(fake, fake_mask).detach()
                gp = gradient_penalty(critic, real, fake_critic, device)
                critic_loss = critic(fake_critic).mean() - critic(real).mean() + float(config["gan"]["gradient_penalty_weight"]) * gp
                opt_c.zero_grad()
                critic_loss.backward()
                opt_c.step()

            z = torch.randn(current_batch, latent_dim, device=device)
            fake, length_probs = generator(
                z,
                length_temperature=float(config["gan"].get("length_temperature", config["generation"].get("temperature", 1.0))),
                return_length=True,
            )
            fake_mask = soft_length_mask(length_probs, min_len, seq_len)
            fake_critic = append_mask_channel(fake, fake_mask)
            predictor_input = soft_esm_features(fake, aa_esm_embeddings, fake_mask) if aa_esm_embeddings is not None else fake * fake_mask.unsqueeze(-1)
            amp_reward = amp_predictor.predict_proba(predictor_input).mean()
            tox_penalty = tox_predictor.predict_proba(predictor_input).mean()
            col_reward = collagen_soft_reward(fake, fake_mask, float(config["gan"].get("collagen_frame_sharpness", 8.0)))
            diversity_reward = batch_diversity_reward(fake, fake_mask)
            len_loss = length_distribution_loss(length_probs, target_lengths)
            comp_penalty = composition_penalty(fake, fake_mask, config)
            complexity_penalty = low_complexity_penalty(fake, fake_mask, config)
            motif_penalty = repeated_motif_penalty(fake, fake_mask, config)
            generator_loss = (
                -critic(fake_critic).mean()
                - float(config["gan"]["amp_reward_weight"]) * amp_reward
                + float(config["gan"]["toxicity_penalty_weight"]) * tox_penalty
                - float(config["gan"]["collagen_reward_weight"]) * col_reward
                - float(config["gan"].get("diversity_reward_weight", 0.0)) * diversity_reward
                + float(config["gan"].get("length_distribution_weight", 0.0)) * len_loss
                + float(config["gan"].get("composition_penalty_weight", 0.0)) * comp_penalty
                + float(config["gan"].get("low_complexity_penalty_weight", 0.0)) * complexity_penalty
                + float(config["gan"].get("repeated_motif_penalty_weight", 0.0)) * motif_penalty
            )
            opt_g.zero_grad()
            generator_loss.backward()
            opt_g.step()
            epoch_stats["generator_loss"].append(float(generator_loss.item()))
            epoch_stats["critic_loss"].append(float(critic_loss.item()))
            epoch_stats["amp_reward"].append(float(amp_reward.item()))
            epoch_stats["toxicity_penalty"].append(float(tox_penalty.item()))
            epoch_stats["collagen_reward"].append(float(col_reward.item()))
            epoch_stats["diversity_reward"].append(float(diversity_reward.item()))
            epoch_stats.setdefault("length_distribution_loss", []).append(float(len_loss.item()))
            epoch_stats["composition_penalty"].append(float(comp_penalty.item()))
            epoch_stats["low_complexity_penalty"].append(float(complexity_penalty.item()))
            epoch_stats["repeated_motif_penalty"].append(float(motif_penalty.item()))
        row = {"epoch": epoch}
        row.update({key: float(np.mean(values)) if values else 0.0 for key, values in epoch_stats.items()})
        history_rows.append(row)
        print(
            f"gan_epoch={epoch:03d} generator_loss={row['generator_loss']:.4f} "
            f"critic_loss={row['critic_loss']:.4f} collagen_reward={row['collagen_reward']:.4f}"
        )
        if epoch % int(config["gan"]["save_every"]) == 0 or epoch == int(config["gan"]["epochs"]):
            torch.save({"model_state": generator.state_dict(), "config": config}, ckpt_dir / "generator.pt")
            pd.DataFrame(history_rows).to_csv(ckpt_dir / "gan_training_history.csv", index=False)
    print(f"Saved GAN generator to {ckpt_dir / 'generator.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    args = parser.parse_args()
    train_gan(load_yaml(args.config))


if __name__ == "__main__":
    main()
