from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()

import pandas as pd

from src.features.collagen_score import collagen_structure_features
from src.features.similarity_search import identity_similarity
from src.utils.metrics import ensure_dir, load_yaml


def safe_to_csv(df: pd.DataFrame, output: Path) -> Path:
    try:
        df.to_csv(output, index=False)
        return output
    except PermissionError:
        fallback = output.with_name(f"{output.stem}_new{output.suffix}")
        df.to_csv(fallback, index=False)
        print(f"{output} is locked; saved to {fallback} instead.")
        return fallback


def ensure_structure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "collagen_motif_score",
        "collagen_gxy_frame",
        "glycine_frame_fraction",
        "gxy_triplet_fraction",
        "xy_proline_fraction",
    }
    if required.issubset(df.columns):
        return df
    structure = pd.DataFrame(
        [collagen_structure_features(sequence).to_dict() for sequence in df["sequence"].astype(str)]
    ).drop(columns=["sequence"])
    for column in structure.columns:
        df[column] = structure[column].to_numpy()
    return df


def _column_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([0.0] * len(df), index=df.index)


def _column_or_one(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([1.0] * len(df), index=df.index)


def diverse_top_k(df: pd.DataFrame, top_k: int, max_selected_similarity: float) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    selected_sequences: list[str] = []
    deferred_rows: list[pd.Series] = []
    for _, row in df.iterrows():
        sequence = str(row["sequence"])
        max_similarity = max((identity_similarity(sequence, selected) for selected in selected_sequences), default=0.0)
        row = row.copy()
        row["max_similarity_to_selected"] = max_similarity
        if max_similarity <= max_selected_similarity or len(selected_rows) == 0:
            selected_rows.append(row)
            selected_sequences.append(sequence)
        else:
            deferred_rows.append(row)
        if len(selected_rows) >= top_k:
            break

    if len(selected_rows) < top_k:
        for row in deferred_rows:
            selected_rows.append(row)
            if len(selected_rows) >= top_k:
                break
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def build_synthesis_shortlist(df: pd.DataFrame, ranking: dict, output_dir: Path) -> pd.DataFrame:
    shortlist_k = int(ranking.get("shortlist_k", 0))
    if shortlist_k <= 0:
        return pd.DataFrame()

    strict = df[
        (df["passed_filter"].astype(bool))
        & (df["collagen_motif_score"] >= float(ranking.get("shortlist_min_collagen_motif_score", 0.0)))
        & (df["glycine_frame_fraction"] >= float(ranking.get("shortlist_min_glycine_frame_fraction", 0.0)))
        & (df["amp_probability"] >= float(ranking.get("shortlist_min_amp_probability", 0.0)))
        & (df["toxicity_probability"] <= float(ranking.get("shortlist_max_toxicity_probability", 1.0)))
        & (df["net_charge"] <= float(ranking.get("shortlist_max_charge", 999.0)))
        & (_column_or_zero(df, "positive_fraction") <= float(ranking.get("shortlist_max_positive_fraction", 1.0)))
        & (_column_or_zero(df, "dipeptide_repeat_fraction") <= float(ranking.get("shortlist_max_dipeptide_repeat_fraction", 1.0)))
        & (_column_or_zero(df, "tripeptide_repeat_fraction") <= float(ranking.get("shortlist_max_tripeptide_repeat_fraction", 1.0)))
        & (_column_or_one(df, "sequence_entropy") >= float(ranking.get("shortlist_min_sequence_entropy", 0.0)))
    ].copy()
    strict["shortlist_strict_pass"] = True
    shortlist = diverse_top_k(
        strict,
        shortlist_k,
        float(ranking.get("shortlist_max_selected_similarity", ranking.get("max_selected_similarity", 0.78))),
    )

    if len(shortlist) < shortlist_k:
        selected = set(shortlist["sequence"].astype(str).tolist()) if not shortlist.empty else set()
        fallback = df[(df["passed_filter"].astype(bool)) & (~df["sequence"].astype(str).isin(selected))].copy()
        fallback["shortlist_strict_pass"] = False
        supplement = diverse_top_k(
            fallback,
            shortlist_k - len(shortlist),
            float(ranking.get("shortlist_max_selected_similarity", ranking.get("max_selected_similarity", 0.78))),
        )
        shortlist = pd.concat([shortlist, supplement], ignore_index=True)

    output = output_dir / f"synthesis_shortlist_{shortlist_k}.csv"
    written = safe_to_csv(shortlist.head(shortlist_k), output)
    print(f"Saved synthesis shortlist {min(shortlist_k, len(shortlist))} candidates to {written}")
    return shortlist.head(shortlist_k)


def rank_candidates(config: dict, input_csv: str | Path | None = None) -> pd.DataFrame:
    input_csv = Path(input_csv or Path(config["paths"]["output_dir"]) / "filtered" / "filtered_candidates.csv")
    df = pd.read_csv(input_csv).drop_duplicates("sequence").reset_index(drop=True)
    df = ensure_structure_columns(df)
    ranking = config.get("ranking", {})
    novelty_threshold = float(ranking.get("novelty_threshold", config["filtering"].get("max_similarity_to_training", 0.95)))
    denominator = max(1e-6, 1.0 - novelty_threshold)
    novelty_penalty = ((df["max_similarity_to_collagen_window"] - novelty_threshold) / denominator).clip(lower=0, upper=1)
    preferred_charge_min = float(ranking.get("preferred_charge_min", config["filtering"].get("min_charge", 2.0)))
    preferred_charge_max = float(ranking.get("preferred_charge_max", config["filtering"].get("max_charge", 5.0)))
    charge_center = (preferred_charge_min + preferred_charge_max) / 2.0
    charge_half_width = max(1.0, (preferred_charge_max - preferred_charge_min) / 2.0)
    charge_score = (1.0 - (df["net_charge"] - charge_center).abs() / charge_half_width).clip(lower=0, upper=1)
    charge_excess = (df["net_charge"] - preferred_charge_max).clip(lower=0) / df["length"].clip(lower=1)
    repetition_penalty = (
        float(ranking.get("repeat_penalty_weight", 0.10)) * _column_or_zero(df, "repeat_fraction")
        + float(ranking.get("dipeptide_repeat_penalty_weight", 0.20)) * _column_or_zero(df, "dipeptide_repeat_fraction")
        + float(ranking.get("tripeptide_repeat_penalty_weight", 0.25)) * _column_or_zero(df, "tripeptide_repeat_fraction")
        + float(ranking.get("longest_run_penalty_weight", 0.10)) * _column_or_zero(df, "longest_run_fraction")
    )
    composition_penalty = (
        float(ranking.get("positive_fraction_penalty_weight", 0.20)) * _column_or_zero(df, "positive_fraction")
        + float(ranking.get("arginine_fraction_penalty_weight", 0.20)) * _column_or_zero(df, "arginine_fraction")
        + float(ranking.get("proline_fraction_penalty_weight", 0.10)) * _column_or_zero(df, "proline_fraction")
        + float(ranking.get("glycine_fraction_penalty_weight", 0.05)) * _column_or_zero(df, "glycine_fraction")
        + float(ranking.get("low_entropy_penalty_weight", 0.25)) * (1.0 - _column_or_one(df, "sequence_entropy"))
    )
    df["final_score"] = (
        float(ranking.get("amp_weight", 0.35)) * df["amp_probability"]
        - float(ranking.get("toxicity_weight", 0.20)) * df["toxicity_probability"]
        + float(ranking.get("charge_weight", 0.08)) * charge_score
        + float(ranking.get("hydrophobic_weight", 0.10)) * df["hydrophobic_fraction"]
        + float(ranking.get("collagen_weight", 0.30)) * df["collagen_motif_score"]
        + float(ranking.get("xy_proline_weight", 0.05)) * df["xy_proline_fraction"]
        - float(ranking.get("novelty_penalty_weight", 0.10)) * novelty_penalty
        - float(ranking.get("charge_excess_penalty_weight", 0.35)) * charge_excess
        - repetition_penalty
        - composition_penalty
    )
    df = df.sort_values(["passed_filter", "final_score"], ascending=[False, False])
    top_k = int(config["generation"]["top_k"])
    ranked = diverse_top_k(df, top_k, float(ranking.get("max_selected_similarity", 0.78)))
    output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "ranked")
    output = output_dir / "ranked_candidates.csv"
    written = safe_to_csv(ranked, output)
    counted_output = output_dir / f"ranked_candidates_top{top_k}.csv"
    if counted_output != written:
        safe_to_csv(ranked, counted_output)
    print(f"Saved top {min(top_k, len(ranked))} ranked candidates to {written}")
    build_synthesis_shortlist(df, ranking, output_dir)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    rank_candidates(load_yaml(args.config), args.input)


if __name__ == "__main__":
    main()
