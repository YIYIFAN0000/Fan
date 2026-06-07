from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()

import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.build_amp_dataset import build_amp_dataset
from src.data.build_collagen_windows import build_collagen_windows
from src.evaluation.classification_metrics import binary_classification_metrics
from src.features.esm2_embeddings import embed_sequences, use_esm_features
from src.features.collagen_score import collagen_structure_features
from src.features.peptide_properties import compute_properties
from src.models.amp_predictor import ESMFeaturePredictor, SequencePredictor
from src.models.toxicity_predictor import ToxicityPredictor
from src.utils.metrics import ensure_dir, load_yaml, set_seed
from src.utils.sequence_utils import AMINO_ACIDS, clean_sequence, is_valid_peptide, one_hot_encode


def heuristic_toxic_label(sequence: str) -> int:
    props = compute_properties(sequence)
    return int(props.hydrophobic_fraction > 0.70 or props.net_charge > 8 or props.cysteine_count > 2)


def load_non_amp_sources(config: dict) -> pd.DataFrame:
    paths = config["paths"]
    data_cfg = config["data"]
    min_len = int(data_cfg["min_len"])
    max_len = int(data_cfg["max_len"])
    raw_dir = Path(paths["raw_dir"])
    candidates: list[Path] = []
    explicit = paths.get("non_amp_negative_csv")
    if explicit and Path(explicit).exists():
        candidates.append(Path(explicit))
    candidates.extend(sorted(raw_dir.glob("non_amp*.csv")))

    frames: list[pd.DataFrame] = []
    for path in candidates:
        df = pd.read_csv(path)
        if "sequence" not in df.columns:
            continue
        df["sequence"] = df["sequence"].map(clean_sequence)
        df = df[df["sequence"].map(lambda seq: is_valid_peptide(seq, min_len, max_len))].copy()
        if df.empty:
            continue
        df["label"] = 0
        df["source_file"] = path.name
        df["source_class"] = "non_amp"
        frames.append(df[["sequence", "source_file", "source_class", "label"]])
    if not frames:
        return pd.DataFrame(columns=["sequence", "source_file", "source_class", "label"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("sequence")


def build_composition_decoys(
    positive_sequences: list[str],
    exclude_sequences: set[str],
    n: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> pd.DataFrame:
    if n <= 0:
        return pd.DataFrame(columns=["sequence", "source_file", "source_class", "label"])

    rng = np.random.default_rng(seed)
    lengths = [len(seq) for seq in positive_sequences if min_len <= len(seq) <= max_len]
    if not lengths:
        lengths = [max_len]

    counts = np.array([1 + sum(seq.count(aa) for seq in positive_sequences) for aa in AMINO_ACIDS], dtype=float)
    probabilities = counts / counts.sum()
    decoys: list[str] = []
    seen = set(exclude_sequences)
    attempts = 0
    max_attempts = max(500, n * 100)
    while len(decoys) < n and attempts < max_attempts:
        attempts += 1
        length = int(rng.choice(lengths))
        sequence = "".join(rng.choice(list(AMINO_ACIDS), size=length, p=probabilities))
        if sequence in seen:
            continue
        if not is_valid_peptide(sequence, min_len, max_len):
            continue
        seen.add(sequence)
        decoys.append(sequence)

    return pd.DataFrame(
        {
            "sequence": decoys,
            "source_file": "composition_decoy",
            "source_class": "composition_decoy",
            "label": 0,
        }
    )


def prepare_train_dataset(config: dict) -> pd.DataFrame:
    positives = build_amp_dataset(config)
    collagen = build_collagen_windows(config)
    positives = positives[["sequence", "source_file", "label"]].copy()
    positives["source_class"] = "amp"

    target_negatives = int(config["data"]["negative_samples"])
    positive_sequences = positives["sequence"].astype(str).tolist()
    exclude = set(positive_sequences) | set(collagen["sequence"].astype(str).tolist())
    negatives = load_non_amp_sources(config)
    negatives = negatives[~negatives["sequence"].isin(set(positive_sequences))].copy()
    if len(negatives) < target_negatives:
        decoys = build_composition_decoys(
            positive_sequences,
            exclude | set(negatives["sequence"].astype(str).tolist()),
            target_negatives - len(negatives),
            int(config["data"]["min_len"]),
            int(config["data"]["max_len"]),
            int(config["seed"]),
        )
        negatives = pd.concat([negatives, decoys], ignore_index=True)
    if len(negatives) > target_negatives:
        negatives = negatives.sample(n=target_negatives, random_state=int(config["seed"]))

    df = pd.concat([positives, negatives], ignore_index=True).drop_duplicates("sequence")
    df["toxicity_label"] = df["sequence"].map(heuristic_toxic_label)
    structure_rows = [collagen_structure_features(sequence).to_dict() for sequence in df["sequence"].astype(str)]
    structure_df = pd.DataFrame(structure_rows).drop(columns=["sequence"])
    df = pd.concat([df.reset_index(drop=True), structure_df.reset_index(drop=True)], axis=1)
    output = Path(config["paths"]["train_dataset_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    collagen.to_csv(config["paths"]["candidate_pool_csv"], index=False)
    return df


def tensorize(df: pd.DataFrame, seq_len: int, target_col: str) -> TensorDataset:
    x = torch.tensor(np.stack([one_hot_encode(seq, seq_len) for seq in df["sequence"]]), dtype=torch.float32)
    y = torch.tensor(df[target_col].to_numpy(dtype="float32"))
    return TensorDataset(x, y)


def feature_tensorize(features: torch.Tensor, df: pd.DataFrame, target_col: str) -> TensorDataset:
    y = torch.tensor(df[target_col].to_numpy(dtype="float32"))
    return TensorDataset(features.float(), y)


def stratified_split_indices(labels: np.ndarray, seed: int, val_fraction: float, test_fraction: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_fraction))) if len(indices) >= 3 else 0
        n_val = max(1, int(round(len(indices) * val_fraction))) if len(indices) - n_test >= 3 else 0
        test_parts.append(indices[:n_test])
        val_parts.append(indices[n_test : n_test + n_val])
        train_parts.append(indices[n_test + n_val :])

    splits = {
        "train": np.concatenate(train_parts) if train_parts else np.array([], dtype=int),
        "val": np.concatenate(val_parts) if val_parts else np.array([], dtype=int),
        "test": np.concatenate(test_parts) if test_parts else np.array([], dtype=int),
    }
    for key, values in splits.items():
        rng.shuffle(values)
        splits[key] = values.astype(int)
    return splits


def subset_dataset(dataset: TensorDataset, indices: np.ndarray) -> TensorDataset:
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return TensorDataset(*[tensor.index_select(0, index_tensor) for tensor in dataset.tensors])


def evaluate_binary_predictor(
    model: nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    if len(dataset) == 0:
        empty = np.array([], dtype=float)
        return 0.0, empty, empty, binary_classification_metrics(empty, empty)
    batch_size = min(512, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            losses.append(float(loss.item()) * len(x))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.cpu().numpy())
    y_true = np.concatenate(targets).astype(int)
    y_score = np.concatenate(probabilities)
    return (
        float(sum(losses) / len(dataset)),
        y_true,
        y_score,
        binary_classification_metrics(y_true, y_score),
    )


def train_binary_predictor(
    model: nn.Module,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    config: dict,
    device: torch.device,
    model_name: str,
) -> tuple[nn.Module, list[dict]]:
    batch_size = min(int(config["predictors"]["batch_size"]), len(train_dataset))
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["predictors"]["learning_rate"]))
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)
    history: list[dict] = []
    for epoch in range(1, int(config["predictors"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(x)
        train_loss, _, _, train_metrics = evaluate_binary_predictor(model, train_dataset, device, loss_fn)
        val_loss, _, _, val_metrics = evaluate_binary_predictor(model, val_dataset, device, loss_fn)
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
            }
        )
        print(
            f"{model_name}_epoch={epoch:03d} loss={total_loss / len(train_dataset):.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_metrics['accuracy']:.4f}"
        )
    return model, history


def train_predictors(config: dict) -> None:
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = prepare_train_dataset(config)
    seq_len = int(config["data"]["seq_len"])
    vocab_size = len(AMINO_ACIDS)
    hidden_dim = int(config["model"]["predictor_hidden_dim"])
    splits = stratified_split_indices(
        df["label"].to_numpy(dtype=int),
        int(config["seed"]),
        float(config["predictors"].get("val_fraction", 0.15)),
        float(config["predictors"].get("test_fraction", 0.15)),
    )

    if use_esm_features(config):
        features = embed_sequences(df["sequence"].astype(str).tolist(), config, cache_name="predictor_train", device=device)
        input_dim = int(features.shape[1])
        config.setdefault("esm", {})["embedding_dim"] = input_dim
        amp_model = ESMFeaturePredictor(input_dim, hidden_dim)
        tox_model = ESMFeaturePredictor(input_dim, hidden_dim)
        amp_dataset = feature_tensorize(features, df, "label")
        tox_dataset = feature_tensorize(features, df, "toxicity_label")
        print(f"Using ESM-2 features for predictors: samples={len(df)} dim={input_dim}")
    else:
        amp_model = SequencePredictor(seq_len, vocab_size, hidden_dim)
        tox_model = ToxicityPredictor(seq_len, vocab_size, hidden_dim)
        amp_dataset = tensorize(df, seq_len, "label")
        tox_dataset = tensorize(df, seq_len, "toxicity_label")

    amp_model, amp_history = train_binary_predictor(
        amp_model,
        subset_dataset(amp_dataset, splits["train"]),
        subset_dataset(amp_dataset, splits["val"]),
        config,
        device,
        "amp_predictor",
    )
    tox_model, tox_history = train_binary_predictor(
        tox_model,
        subset_dataset(tox_dataset, splits["train"]),
        subset_dataset(tox_dataset, splits["val"]),
        config,
        device,
        "toxicity_predictor",
    )

    output_dir = ensure_dir(config["paths"]["predictor_dir"])
    loss_fn = nn.BCEWithLogitsLoss()
    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for model_name, model, dataset in (
        ("amp_predictor", amp_model, amp_dataset),
        ("toxicity_predictor", tox_model, tox_dataset),
    ):
        for split_name, indices in splits.items():
            split_dataset = subset_dataset(dataset, indices)
            loss, y_true, y_score, metrics = evaluate_binary_predictor(model, split_dataset, device, loss_fn)
            metrics_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "n": len(indices),
                    "loss": loss,
                    **metrics,
                }
            )
            for index, label, probability in zip(indices, y_true, y_score):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "sequence": df.iloc[int(index)]["sequence"],
                        "label": int(label),
                        "probability": float(probability),
                    }
                )

    pd.DataFrame([*amp_history, *tox_history]).to_csv(output_dir / "predictor_training_history.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "predictor_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictor_predictions.csv", index=False)
    torch.save({"model_state": amp_model.state_dict(), "config": config}, output_dir / "amp_predictor.pt")
    torch.save({"model_state": tox_model.state_dict(), "config": config}, output_dir / "toxicity_predictor.pt")
    print(f"Saved predictors to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    args = parser.parse_args()
    train_predictors(load_yaml(args.config))


if __name__ == "__main__":
    main()
