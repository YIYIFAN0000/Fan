from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()

import pandas as pd
import torch

from src.features.collagen_score import collagen_structure_features
from src.features.esm2_embeddings import embed_sequences, esm_feature_dim, use_esm_features
from src.features.peptide_properties import compute_properties, passes_rule_filter
from src.features.similarity_search import max_similarity
from src.models.amp_predictor import ESMFeaturePredictor, SequencePredictor
from src.models.toxicity_predictor import ToxicityPredictor
from src.utils.metrics import ensure_dir, load_yaml
from src.utils.sequence_utils import AMINO_ACIDS, one_hot_encode


def load_predictor(path: str | Path, model_cls, config: dict, device: torch.device):
    if use_esm_features(config):
        model = ESMFeaturePredictor(
            esm_feature_dim(config),
            int(config["model"]["predictor_hidden_dim"]),
        ).to(device)
    else:
        model = model_cls(
            int(config["data"]["seq_len"]),
            len(AMINO_ACIDS),
            int(config["model"]["predictor_hidden_dim"]),
        ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_probability(model, sequence: str, config: dict, device: torch.device) -> float:
    x = torch.tensor(one_hot_encode(sequence, int(config["data"]["seq_len"])), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(model.predict_proba(x).cpu().item())


def filter_candidates(config: dict, input_csv: str | Path | None = None) -> pd.DataFrame:
    input_csv = Path(input_csv or Path(config["paths"]["output_dir"]) / "generated" / "generated_peptides.csv")
    df = pd.read_csv(input_csv).drop_duplicates("sequence").reset_index(drop=True)
    collagen_df = pd.read_csv(config["paths"]["collagen_windows_csv"])
    similarity_reference_limit = int(config.get("filtering", {}).get("similarity_reference_limit", 0))
    if similarity_reference_limit > 0 and len(collagen_df) > similarity_reference_limit:
        collagen_df = collagen_df.sample(n=similarity_reference_limit, random_state=int(config["seed"]))
    collagen_refs = collagen_df["sequence"].astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor_dir = Path(config["paths"]["predictor_dir"])
    amp_model = load_predictor(predictor_dir / "amp_predictor.pt", SequencePredictor, config, device)
    tox_model = load_predictor(predictor_dir / "toxicity_predictor.pt", ToxicityPredictor, config, device)
    sequences = df["sequence"].astype(str).tolist()
    if use_esm_features(config):
        features = embed_sequences(sequences, config, cache_name="filter_candidates", device=device).to(device)
        with torch.no_grad():
            amp_probabilities = amp_model.predict_proba(features).cpu().tolist()
            tox_probabilities = tox_model.predict_proba(features).cpu().tolist()
    else:
        amp_probabilities = [predict_probability(amp_model, sequence, config, device) for sequence in sequences]
        tox_probabilities = [predict_probability(tox_model, sequence, config, device) for sequence in sequences]

    rows: list[dict] = []
    for sequence, amp_probability, tox_probability in zip(sequences, amp_probabilities, tox_probabilities):
        props = compute_properties(sequence).to_dict()
        props.update(collagen_structure_features(sequence).to_dict())
        props["amp_probability"] = float(amp_probability)
        props["toxicity_probability"] = float(tox_probability)
        props["max_similarity_to_collagen_window"] = max_similarity(sequence, collagen_refs)
        props["rule_pass"] = passes_rule_filter(sequence, config["filtering"])
        props["passed_filter"] = (
            props["rule_pass"]
            and props["amp_probability"] >= float(config["filtering"]["min_amp_probability"])
            and props["toxicity_probability"] <= float(config["filtering"]["max_toxicity_probability"])
            and props["collagen_motif_score"] >= float(config["filtering"].get("min_collagen_motif_score", 0.0))
            and props["glycine_frame_fraction"] >= float(config["filtering"].get("min_glycine_frame_fraction", 0.0))
            and props["max_similarity_to_collagen_window"] <= float(config["filtering"]["max_similarity_to_training"])
        )
        rows.append(props)
    out = pd.DataFrame(rows)
    output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "filtered")
    output = output_dir / "filtered_candidates.csv"
    out.to_csv(output, index=False)
    print(f"Saved {int(out['passed_filter'].sum())}/{len(out)} filtered candidates to {output}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    filter_candidates(load_yaml(args.config), args.input)


if __name__ == "__main__":
    main()
