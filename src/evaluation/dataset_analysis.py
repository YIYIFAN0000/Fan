from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.peptide_properties import compute_properties
from src.utils.metrics import ensure_dir
from src.utils.sequence_utils import AMINO_ACIDS, clean_sequence


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save_figure(fig, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    return paths


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


def add_split_column(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    out["split"] = ""
    splits = stratified_split_indices(
        out["label"].to_numpy(dtype=int),
        int(config["seed"]),
        float(config["predictors"].get("val_fraction", 0.15)),
        float(config["predictors"].get("test_fraction", 0.15)),
    )
    for split_name, indices in splits.items():
        out.loc[indices, "split"] = split_name
    return out


def enrich_properties(df: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "net_charge",
        "hydrophobic_fraction",
        "positive_fraction",
        "sequence_entropy",
        "dipeptide_repeat_fraction",
        "tripeptide_repeat_fraction",
    }
    if needed.issubset(df.columns):
        return df
    rows = [compute_properties(sequence).to_dict() for sequence in df["sequence"].astype(str)]
    props = pd.DataFrame(rows).drop(columns=["sequence"])
    merged = df.copy()
    for column in props.columns:
        if column not in merged.columns:
            merged[column] = props[column].to_numpy()
    return merged


def aa_composition(sequences: list[str]) -> dict[str, float]:
    counts = Counter()
    total = 0
    for sequence in sequences:
        cleaned = clean_sequence(sequence)
        counts.update(cleaned)
        total += len(cleaned)
    if total == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}
    return {aa: counts[aa] / total for aa in AMINO_ACIDS}


def aa_count_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_cols, keys)}
        sequences = group["sequence"].astype(str).tolist()
        counts = Counter()
        total = 0
        for sequence in sequences:
            cleaned = clean_sequence(sequence)
            counts.update(cleaned)
            total += len(cleaned)
        row["n_sequences"] = len(group)
        row["total_residues"] = total
        for aa in AMINO_ACIDS:
            row[f"{aa}_count"] = counts[aa]
            row[f"{aa}_fraction"] = counts[aa] / total if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary_tables(
    train_df: pd.DataFrame,
    collagen_df: pd.DataFrame,
    output_dir: Path,
    raw_dir: Path,
) -> dict[str, Path]:
    tables_dir = ensure_dir(output_dir / "tables")
    paths: dict[str, Path] = {}

    overview_rows = [
        {"dataset": "train_dataset", "n_sequences": len(train_df), "n_unique": train_df["sequence"].nunique()},
        {
            "dataset": "positive_amp",
            "n_sequences": int((train_df["label"] == 1).sum()),
            "n_unique": train_df.loc[train_df["label"] == 1, "sequence"].nunique(),
        },
        {
            "dataset": "negative",
            "n_sequences": int((train_df["label"] == 0).sum()),
            "n_unique": train_df.loc[train_df["label"] == 0, "sequence"].nunique(),
        },
        {
            "dataset": "collagen_windows",
            "n_sequences": len(collagen_df),
            "n_unique": collagen_df["sequence"].nunique() if "sequence" in collagen_df.columns else 0,
        },
        {
            "dataset": "raw_user_dir_available",
            "n_sequences": int(raw_dir.exists()),
            "n_unique": int(raw_dir.exists()),
        },
    ]
    overview = pd.DataFrame(overview_rows)
    paths["dataset_overview"] = tables_dir / "dataset_overview.csv"
    overview.to_csv(paths["dataset_overview"], index=False)

    source_counts = (
        train_df.groupby(["source_class", "source_file", "label"], dropna=False)
        .size()
        .reset_index(name="n_sequences")
        .sort_values(["source_class", "source_file", "label"])
    )
    paths["source_counts"] = tables_dir / "source_counts.csv"
    source_counts.to_csv(paths["source_counts"], index=False)

    split_counts = (
        train_df.groupby(["split", "label", "source_class"], dropna=False)
        .size()
        .reset_index(name="n_sequences")
        .sort_values(["split", "label", "source_class"])
    )
    paths["split_counts"] = tables_dir / "split_counts.csv"
    split_counts.to_csv(paths["split_counts"], index=False)

    length_summary = (
        train_df.groupby(["split", "label"], dropna=False)["length"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    paths["length_summary_by_split_label"] = tables_dir / "length_summary_by_split_label.csv"
    length_summary.to_csv(paths["length_summary_by_split_label"], index=False)

    length_distribution = (
        train_df.groupby(["split", "label", "length"], dropna=False)
        .size()
        .reset_index(name="n_sequences")
        .sort_values(["split", "label", "length"])
    )
    paths["length_distribution"] = tables_dir / "length_distribution_by_split_label.csv"
    length_distribution.to_csv(paths["length_distribution"], index=False)

    aa_by_label = aa_count_table(train_df, ["label"])
    paths["aa_composition_by_label"] = tables_dir / "aa_composition_by_label.csv"
    aa_by_label.to_csv(paths["aa_composition_by_label"], index=False)

    aa_by_split_label = aa_count_table(train_df, ["split", "label"])
    paths["aa_composition_by_split_label"] = tables_dir / "aa_composition_by_split_label.csv"
    aa_by_split_label.to_csv(paths["aa_composition_by_split_label"], index=False)

    properties = [
        "net_charge",
        "charge_density",
        "hydrophobic_fraction",
        "positive_fraction",
        "glycine_fraction",
        "proline_fraction",
        "sequence_entropy",
        "dipeptide_repeat_fraction",
        "tripeptide_repeat_fraction",
        "collagen_motif_score",
        "glycine_frame_fraction",
    ]
    available = [column for column in properties if column in train_df.columns]
    property_summary = (
        train_df.groupby(["split", "label"], dropna=False)[available]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    property_summary.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in property_summary.columns
    ]
    paths["property_summary_by_split_label"] = tables_dir / "property_summary_by_split_label.csv"
    property_summary.to_csv(paths["property_summary_by_split_label"], index=False)

    if not collagen_df.empty:
        collagen_summary = (
            collagen_df.groupby(["length"], dropna=False)
            .agg(
                n_sequences=("sequence", "count"),
                collagen_motif_score_mean=("collagen_motif_score", "mean"),
                glycine_frame_fraction_mean=("glycine_frame_fraction", "mean"),
                xy_proline_fraction_mean=("xy_proline_fraction", "mean"),
            )
            .reset_index()
        )
        paths["collagen_window_length_summary"] = tables_dir / "collagen_window_length_summary.csv"
        collagen_summary.to_csv(paths["collagen_window_length_summary"], index=False)

        if "record_id" in collagen_df.columns:
            collagen_sources = (
                collagen_df.groupby("record_id", dropna=False)
                .size()
                .reset_index(name="n_windows")
                .sort_values("n_windows", ascending=False)
            )
            paths["collagen_source_counts"] = tables_dir / "collagen_source_counts.csv"
            collagen_sources.to_csv(paths["collagen_source_counts"], index=False)

    return paths


def _label_name(label: int | str) -> str:
    return "positive AMP" if int(label) == 1 else "negative / decoy"


def save_figures(train_df: pd.DataFrame, collagen_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    figures_dir = ensure_dir(output_dir / "figures")
    written: list[Path] = []
    plt = _pyplot()

    source_counts = (
        train_df.groupby(["source_class", "label"], dropna=False)
        .size()
        .reset_index(name="n_sequences")
        .sort_values("n_sequences", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    source_counts["source_label"] = source_counts.apply(
        lambda row: f"{row['source_class']} ({_label_name(row['label'])})",
        axis=1,
    )
    ax.barh(source_counts["source_label"], source_counts["n_sequences"], color=["#4C78A8", "#F58518"][: len(source_counts)])
    ax.set_xlabel("Sequences")
    ax.set_title("Training Data Source Composition")
    ax.invert_yaxis()
    fig.tight_layout()
    written.extend(_save_figure(fig, figures_dir / "source_composition"))
    plt.close(fig)

    label_counts = train_df.groupby("label").size().reindex([1, 0], fill_value=0)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar([_label_name(label) for label in label_counts.index], label_counts.values, color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Sequences")
    ax.set_title("Training Dataset Label Balance")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    written.extend(_save_figure(fig, figures_dir / "dataset_label_balance"))
    plt.close(fig)

    split_counts = train_df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(["train", "val", "test"])
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = np.arange(len(split_counts.index))
    width = 0.35
    ax.bar(x - width / 2, split_counts.get(1, pd.Series(0, index=split_counts.index)), width, label="Positive AMP")
    ax.bar(x + width / 2, split_counts.get(0, pd.Series(0, index=split_counts.index)), width, label="Negative / decoy")
    ax.set_xticks(x)
    ax.set_xticklabels(split_counts.index)
    ax.set_ylabel("Sequences")
    ax.set_title("Predictor Split Composition")
    ax.legend(frameon=False)
    fig.tight_layout()
    written.extend(_save_figure(fig, figures_dir / "split_label_counts"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bins = np.arange(int(train_df["length"].min()), int(train_df["length"].max()) + 2) - 0.5
    for label, color in ((1, "#4C78A8"), (0, "#F58518")):
        values = train_df.loc[train_df["label"] == label, "length"]
        ax.hist(values, bins=bins, alpha=0.60, label=_label_name(label), color=color)
    ax.set_xlabel("Peptide length")
    ax.set_ylabel("Sequences")
    ax.set_title("Length Distribution by Label")
    ax.legend(frameon=False)
    fig.tight_layout()
    written.extend(_save_figure(fig, figures_dir / "length_distribution_by_label"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), sharey=True)
    for ax, split in zip(axes, ["train", "val", "test"]):
        subset = train_df[train_df["split"] == split]
        bins = np.arange(int(train_df["length"].min()), int(train_df["length"].max()) + 2) - 0.5
        for label, color in ((1, "#4C78A8"), (0, "#F58518")):
            ax.hist(subset.loc[subset["label"] == label, "length"], bins=bins, alpha=0.60, color=color, label=_label_name(label))
        ax.set_title(split)
        ax.set_xlabel("Length")
    axes[0].set_ylabel("Sequences")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    written.extend(_save_figure(fig, figures_dir / "length_distribution_by_split"))
    plt.close(fig)

    aa_label_rows = []
    for label in [1, 0]:
        row = aa_composition(train_df.loc[train_df["label"] == label, "sequence"].astype(str).tolist())
        row["label"] = _label_name(label)
        aa_label_rows.append(row)
    aa_label = pd.DataFrame(aa_label_rows).set_index("label")[list(AMINO_ACIDS)]
    fig, ax = plt.subplots(figsize=(8.0, 3.3))
    x = np.arange(len(AMINO_ACIDS))
    width = 0.38
    ax.bar(x - width / 2, aa_label.loc["positive AMP"].values, width, label="Positive AMP", color="#4C78A8")
    ax.bar(x + width / 2, aa_label.loc["negative / decoy"].values, width, label="Negative / decoy", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(list(AMINO_ACIDS))
    ax.set_ylabel("Residue fraction")
    ax.set_title("Amino Acid Composition by Label")
    ax.legend(frameon=False)
    fig.tight_layout()
    written.extend(_save_figure(fig, figures_dir / "amino_acid_composition_by_label"))
    plt.close(fig)

    aa_delta = aa_label.loc["positive AMP"] - aa_label.loc["negative / decoy"]
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    colors = ["#4C78A8" if value >= 0 else "#F58518" for value in aa_delta.values]
    ax.bar(np.arange(len(AMINO_ACIDS)), aa_delta.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(AMINO_ACIDS)))
    ax.set_xticklabels(list(AMINO_ACIDS))
    ax.set_ylabel("Positive - negative fraction")
    ax.set_title("Amino Acid Fraction Difference")
    fig.tight_layout()
    written.extend(_save_figure(fig, figures_dir / "amino_acid_fraction_difference"))
    plt.close(fig)

    heat_rows = []
    heat_labels = []
    for split in ["train", "val", "test"]:
        for label in [1, 0]:
            subset = train_df[(train_df["split"] == split) & (train_df["label"] == label)]
            heat_rows.append([aa_composition(subset["sequence"].astype(str).tolist())[aa] for aa in AMINO_ACIDS])
            heat_labels.append(f"{split} {_label_name(label)}")
    heat = np.array(heat_rows)
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    image = ax.imshow(heat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(AMINO_ACIDS)))
    ax.set_xticklabels(list(AMINO_ACIDS))
    ax.set_yticks(np.arange(len(heat_labels)))
    ax.set_yticklabels(heat_labels)
    ax.set_title("Amino Acid Fractions by Split and Label")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Residue fraction")
    fig.tight_layout()
    written.extend(_save_figure(fig, figures_dir / "amino_acid_composition_heatmap"))
    plt.close(fig)

    property_columns = [
        ("net_charge", "Net charge"),
        ("hydrophobic_fraction", "Hydrophobic fraction"),
        ("positive_fraction", "Positive fraction"),
        ("sequence_entropy", "Sequence entropy"),
        ("collagen_motif_score", "Collagen motif score"),
    ]
    property_columns = [(column, label) for column, label in property_columns if column in train_df.columns]
    fig, axes = plt.subplots(1, len(property_columns), figsize=(2.2 * len(property_columns), 3.2))
    if len(property_columns) == 1:
        axes = [axes]
    for ax, (column, title) in zip(axes, property_columns):
        positive = train_df.loc[train_df["label"] == 1, column].dropna()
        negative = train_df.loc[train_df["label"] == 0, column].dropna()
        ax.boxplot(
            [positive, negative],
            labels=["Positive", "Negative"],
            widths=0.55,
            patch_artist=True,
            boxprops={"facecolor": "#9DB7D0", "edgecolor": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.0},
            flierprops={"markersize": 2.0, "alpha": 0.30},
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    written.extend(_save_figure(fig, figures_dir / "property_distributions_by_label"))
    plt.close(fig)

    if not collagen_df.empty and "length" in collagen_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
        bins = np.arange(int(collagen_df["length"].min()), int(collagen_df["length"].max()) + 2) - 0.5
        axes[0].hist(collagen_df["length"], bins=bins, color="#54A24B", alpha=0.75)
        axes[0].set_xlabel("Collagen window length")
        axes[0].set_ylabel("Windows")
        axes[0].set_title("Collagen Window Lengths")
        axes[1].hist(collagen_df["collagen_motif_score"], bins=np.linspace(0, 1, 31), color="#54A24B", alpha=0.75)
        axes[1].set_xlabel("Collagen motif score")
        axes[1].set_ylabel("Windows")
        axes[1].set_title("Collagen Motif Score")
        fig.tight_layout()
        written.extend(_save_figure(fig, figures_dir / "collagen_window_distributions"))
        plt.close(fig)

        if "record_id" in collagen_df.columns:
            top_sources = (
                collagen_df.groupby("record_id")
                .size()
                .sort_values(ascending=False)
                .head(10)
                .sort_values(ascending=True)
            )
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            labels = [str(label).split(" OS=")[0].replace("sp|", "")[:55] for label in top_sources.index]
            ax.barh(labels, top_sources.values, color="#54A24B", alpha=0.8)
            ax.set_xlabel("Windows")
            ax.set_title("Top Collagen FASTA Sources")
            fig.tight_layout()
            written.extend(_save_figure(fig, figures_dir / "collagen_source_top10"))
            plt.close(fig)

    pd.DataFrame({"figure": [str(path) for path in written]}).to_csv(figures_dir / "dataset_figure_manifest.csv", index=False)
    return written


def analyze_dataset(config: dict) -> dict[str, list[Path] | dict[str, Path]]:
    output_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "dataset_analysis")
    train_csv = Path(config["paths"]["train_dataset_csv"])
    collagen_csv = Path(config["paths"]["collagen_windows_csv"])
    if not train_csv.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_csv}")

    train_df = pd.read_csv(train_csv)
    train_df = add_split_column(train_df, config)
    train_df = enrich_properties(train_df)
    collagen_df = pd.read_csv(collagen_csv) if collagen_csv.exists() else pd.DataFrame()

    annotated_path = output_dir / "tables" / "train_dataset_with_splits.csv"
    ensure_dir(annotated_path.parent)
    train_df.to_csv(annotated_path, index=False)

    table_paths = write_summary_tables(train_df, collagen_df, output_dir, Path(config["paths"]["raw_dir"]))
    table_paths["train_dataset_with_splits"] = annotated_path
    figure_paths = save_figures(train_df, collagen_df, output_dir)
    return {"tables": table_paths, "figures": figure_paths}
