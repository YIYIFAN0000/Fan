from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.classification_metrics import precision_recall_curve_points, roc_curve_points
from src.utils.metrics import ensure_dir


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


def _read_existing_csv(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def save_predictor_figures(predictor_dir: str | Path, figure_dir: str | Path) -> list[Path]:
    predictor_dir = Path(predictor_dir)
    figure_dir = ensure_dir(figure_dir)
    metrics_csv = predictor_dir / "predictor_metrics.csv"
    history_csv = predictor_dir / "predictor_training_history.csv"
    predictions_csv = predictor_dir / "predictor_predictions.csv"
    written: list[Path] = []
    plt = _pyplot()

    if metrics_csv.exists():
        metrics = pd.read_csv(metrics_csv)
        test_metrics = metrics[metrics["split"] == "test"].copy()
        if not test_metrics.empty:
            metric_names = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
            labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "PR-AUC"]
            x = np.arange(len(metric_names))
            models = test_metrics["model"].tolist()
            width = min(0.35, 0.8 / max(1, len(models)))
            fig, ax = plt.subplots(figsize=(7.0, 3.2))
            for index, (_, row) in enumerate(test_metrics.iterrows()):
                offsets = x + (index - (len(models) - 1) / 2) * width
                ax.bar(offsets, [row[name] for name in metric_names], width=width, label=row["model"])
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.legend(frameon=False, ncol=2)
            ax.set_title("Predictor Test Performance")
            fig.tight_layout()
            written.extend(_save_figure(fig, figure_dir / "predictor_test_metrics"))
            plt.close(fig)

    if history_csv.exists():
        history = pd.read_csv(history_csv)
        if not history.empty:
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=False)
            for model_name, group in history.groupby("model"):
                axes[0].plot(group["epoch"], group["train_loss"], label=f"{model_name} train")
                axes[0].plot(group["epoch"], group["val_loss"], linestyle="--", label=f"{model_name} val")
                axes[1].plot(group["epoch"], group["val_accuracy"], label=f"{model_name} accuracy")
                axes[1].plot(group["epoch"], group["val_f1"], linestyle="--", label=f"{model_name} F1")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("BCE loss")
            axes[0].set_title("Training Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Validation score")
            axes[1].set_ylim(0, 1.05)
            axes[1].set_title("Validation Performance")
            for ax in axes:
                ax.legend(frameon=False)
            fig.tight_layout()
            written.extend(_save_figure(fig, figure_dir / "predictor_training_curves"))
            plt.close(fig)

    if predictions_csv.exists():
        predictions = pd.read_csv(predictions_csv)
        test_predictions = predictions[predictions["split"] == "test"].copy()
        if not test_predictions.empty:
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
            for model_name, group in test_predictions.groupby("model"):
                y_true = group["label"].to_numpy(dtype=int)
                y_score = group["probability"].to_numpy(dtype=float)
                fpr, tpr, roc_auc = roc_curve_points(y_true, y_score)
                recall, precision, pr_auc = precision_recall_curve_points(y_true, y_score)
                axes[0].plot(fpr, tpr, label=f"{model_name} AUC={roc_auc:.3f}")
                axes[1].plot(recall, precision, label=f"{model_name} AUC={pr_auc:.3f}")
            axes[0].plot([0, 1], [0, 1], color="0.7", linewidth=1, linestyle="--")
            axes[0].set_xlabel("False positive rate")
            axes[0].set_ylabel("True positive rate")
            axes[0].set_title("ROC Curve")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall Curve")
            for ax in axes:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.05)
                ax.legend(frameon=False)
            fig.tight_layout()
            written.extend(_save_figure(fig, figure_dir / "predictor_roc_pr_curves"))
            plt.close(fig)
    return written


def save_gan_training_figure(checkpoint_dir: str | Path, figure_dir: str | Path) -> list[Path]:
    checkpoint_dir = Path(checkpoint_dir)
    history_csv = checkpoint_dir / "gan_training_history.csv"
    if not history_csv.exists():
        return []
    history = pd.read_csv(history_csv)
    if history.empty:
        return []

    figure_dir = ensure_dir(figure_dir)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    axes[0].plot(history["epoch"], history["generator_loss"], label="Generator")
    axes[0].plot(history["epoch"], history["critic_loss"], label="Critic")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("WGAN-GP Loss")
    for column, label in (
        ("amp_reward", "AMP reward"),
        ("toxicity_penalty", "Toxicity penalty"),
        ("collagen_reward", "Collagen reward"),
        ("diversity_reward", "Diversity reward"),
    ):
        if column in history.columns:
            axes[1].plot(history["epoch"], history[column], label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reward / penalty")
    axes[1].set_title("Generator Objectives")
    for column, label in (
        ("composition_penalty", "Composition"),
        ("low_complexity_penalty", "Low complexity"),
        ("repeated_motif_penalty", "Repeated motif"),
        ("length_distribution_loss", "Length KL"),
    ):
        if column in history.columns:
            axes[2].plot(history["epoch"], history[column], label=label)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Penalty")
    axes[2].set_title("Regularization")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    paths = _save_figure(fig, figure_dir / "gan_training_curves")
    plt.close(fig)
    return paths


def save_candidate_quality_figures(config: dict, figure_dir: str | Path) -> list[Path]:
    output_dir = Path(config["paths"]["output_dir"])
    filtered_csv = output_dir / "filtered" / "filtered_candidates.csv"
    top_k = int(config.get("generation", {}).get("top_k", 100))
    shortlist_k = int(config.get("ranking", {}).get("shortlist_k", 50))
    if not filtered_csv.exists():
        return []

    filtered = pd.read_csv(filtered_csv)
    ranked = _read_existing_csv(
        [
            output_dir / "ranked" / f"ranked_candidates_top{top_k}.csv",
            output_dir / "ranked" / "ranked_candidates_new.csv",
            output_dir / "ranked" / "ranked_candidates.csv",
        ]
    )
    shortlist = _read_existing_csv(
        [
            output_dir / "ranked" / f"synthesis_shortlist_{shortlist_k}.csv",
            output_dir / "ranked" / "synthesis_shortlist_50.csv",
        ]
    )
    if filtered.empty:
        return []

    figure_dir = ensure_dir(figure_dir)
    plt = _pyplot()
    written: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))
    for ax, column, label, threshold in (
        (
            axes[0],
            "amp_probability",
            "AMP probability",
            float(config["filtering"].get("min_amp_probability", 0.0)),
        ),
        (
            axes[1],
            "toxicity_probability",
            "Toxicity probability",
            float(config["filtering"].get("max_toxicity_probability", 1.0)),
        ),
        (
            axes[2],
            "collagen_motif_score",
            "Collagen motif score",
            float(config["filtering"].get("min_collagen_motif_score", 0.0)),
        ),
    ):
        bins = np.linspace(0, 1, 31)
        ax.hist(filtered[column], bins=bins, color="#4C78A8", alpha=0.55, label="Generated")
        if not ranked.empty and column in ranked.columns:
            ax.hist(ranked[column], bins=np.linspace(0, 1, 21), color="#F58518", alpha=0.65, label="Ranked")
        if not shortlist.empty and column in shortlist.columns:
            ax.hist(
                shortlist[column],
                bins=np.linspace(0, 1, 16),
                color="#E45756",
                alpha=0.60,
                label="Shortlist",
            )
        ax.axvline(threshold, color="black", linewidth=1, linestyle="--")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    written.extend(_save_figure(fig, figure_dir / "candidate_quality_distributions"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    colors = filtered["collagen_motif_score"] if "collagen_motif_score" in filtered.columns else "0.4"
    scatter = ax.scatter(
        filtered["toxicity_probability"],
        filtered["amp_probability"],
        c=colors,
        cmap="viridis",
        s=18,
        alpha=0.75,
        linewidths=0,
    )
    ax.axhline(float(config["filtering"].get("min_amp_probability", 0.0)), color="black", linewidth=1, linestyle="--")
    ax.axvline(float(config["filtering"].get("max_toxicity_probability", 1.0)), color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Toxicity probability")
    ax.set_ylabel("AMP probability")
    ax.set_title("Candidate Activity-Toxicity Tradeoff")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Collagen motif score")
    fig.tight_layout()
    written.extend(_save_figure(fig, figure_dir / "candidate_activity_toxicity_structure"))
    plt.close(fig)

    if "glycine_frame_fraction" in filtered.columns:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.hist(
            filtered["glycine_frame_fraction"],
            bins=np.linspace(0, 1, 21),
            color="#54A24B",
            alpha=0.6,
            label="Generated",
        )
        if not ranked.empty and "glycine_frame_fraction" in ranked.columns:
            ax.hist(
                ranked["glycine_frame_fraction"],
                bins=np.linspace(0, 1, 16),
                color="#E45756",
                alpha=0.65,
                label="Ranked",
            )
        if not shortlist.empty and "glycine_frame_fraction" in shortlist.columns:
            ax.hist(
                shortlist["glycine_frame_fraction"],
                bins=np.linspace(0, 1, 16),
                color="#F58518",
                alpha=0.60,
                label="Shortlist",
            )
        ax.axvline(
            float(config["filtering"].get("min_glycine_frame_fraction", 0.0)),
            color="black",
            linewidth=1,
            linestyle="--",
        )
        ax.set_xlabel("Best-frame Gly fraction")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1.0)
        ax.set_title("G-X-Y Frame Preservation")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.extend(_save_figure(fig, figure_dir / "candidate_gxy_frame_preservation"))
        plt.close(fig)

    if "length" in filtered.columns:
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        length_sources = [filtered["length"]]
        if not ranked.empty and "length" in ranked.columns:
            length_sources.append(ranked["length"])
        if not shortlist.empty and "length" in shortlist.columns:
            length_sources.append(shortlist["length"])
        min_len = int(min(series.min() for series in length_sources))
        max_len = int(max(series.max() for series in length_sources))
        bins = np.arange(min_len, max_len + 2) - 0.5
        ax.hist(filtered["length"], bins=bins, color="#4C78A8", alpha=0.55, label="Generated")
        if not ranked.empty and "length" in ranked.columns:
            ax.hist(ranked["length"], bins=bins, color="#F58518", alpha=0.65, label="Ranked")
        if not shortlist.empty and "length" in shortlist.columns:
            ax.hist(shortlist["length"], bins=bins, color="#E45756", alpha=0.55, label="Shortlist")
        ax.set_xlabel("Peptide length")
        ax.set_ylabel("Count")
        ax.set_xticks(np.arange(min_len, max_len + 1, max(1, (max_len - min_len) // 8 or 1)))
        ax.set_title("Variable-Length Candidate Distribution")
        ax.legend(frameon=False)
        fig.tight_layout()
        written.extend(_save_figure(fig, figure_dir / "candidate_length_distribution"))
        plt.close(fig)

    complexity_columns = [
        ("net_charge", "Net charge"),
        ("positive_fraction", "Positive fraction"),
        ("sequence_entropy", "Sequence entropy"),
        ("dipeptide_repeat_fraction", "Dipeptide repeat"),
        ("tripeptide_repeat_fraction", "Tripeptide repeat"),
    ]
    if all(column in filtered.columns for column, _ in complexity_columns):
        fig, axes = plt.subplots(1, len(complexity_columns), figsize=(10.0, 2.8))
        for ax, (column, label) in zip(axes, complexity_columns):
            generated_values = filtered[column].dropna()
            ranked_values = ranked[column].dropna() if not ranked.empty and column in ranked.columns else pd.Series(dtype=float)
            shortlist_values = (
                shortlist[column].dropna()
                if not shortlist.empty and column in shortlist.columns
                else pd.Series(dtype=float)
            )
            ax.boxplot(
                [generated_values, ranked_values, shortlist_values],
                labels=["Generated", "Ranked", "Shortlist"],
                widths=0.55,
                patch_artist=True,
                boxprops={"facecolor": "#9DB7D0", "edgecolor": "black", "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"linewidth": 0.8},
                capprops={"linewidth": 0.8},
                flierprops={"markersize": 2.0, "alpha": 0.35},
            )
            ax.set_title(label)
            ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        written.extend(_save_figure(fig, figure_dir / "candidate_composition_complexity"))
        plt.close(fig)
    return written


def write_all_figures(config: dict) -> list[Path]:
    figure_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "figures")
    written: list[Path] = []
    written.extend(save_predictor_figures(config["paths"]["predictor_dir"], figure_dir))
    written.extend(save_gan_training_figure(config["paths"]["checkpoint_dir"], figure_dir))
    written.extend(save_candidate_quality_figures(config, figure_dir))
    if written:
        pd.DataFrame({"figure": [str(path) for path in written]}).to_csv(figure_dir / "figure_manifest.csv", index=False)
    return written
