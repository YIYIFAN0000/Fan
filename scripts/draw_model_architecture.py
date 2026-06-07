from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"


COLORS = {
    "data": "#DCE7F3",
    "encoding": "#D7F1F0",
    "generator": "#E7DDF6",
    "critic": "#F5D9BE",
    "predictor": "#DDEEDC",
    "screening": "#F7EBC5",
    "final": "#123C69",
    "final2": "#0F4C3A",
    "line": "#3F4650",
    "muted": "#6B7280",
}


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#4B5563",
    textcolor: str = "#111827",
    fontsize: float = 8.2,
    weight: str = "normal",
    radius: float = 0.08,
    linewidth: float = 0.85,
    wrap_width: int = 22,
) -> FancyBboxPatch:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    label = "\n".join(wrap(text, wrap_width, break_long_words=False))
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        weight=weight,
        linespacing=1.15,
    )
    return patch


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    lw: float = 1.1,
    style: str = "-",
    mutation_scale: float = 10,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=style,
        shrinkA=3,
        shrinkB=3,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def add_label(ax, x: float, y: float, text: str, color: str = COLORS["muted"]) -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=7.6, color=color)


def add_layer_band(ax, x: float, width: float, title: str) -> None:
    ax.plot([x, x + width], [0.58, 0.58], color="#CBD5E1", lw=0.8)
    ax.text(
        x + width / 2,
        0.34,
        title,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#4B5563",
    )


def draw() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(17.2, 7.8), dpi=300)
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 7.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        0.35,
        7.46,
        "Collagen-constrained adversarial generation of antimicrobial peptide candidates",
        ha="left",
        va="center",
        fontsize=13.5,
        weight="bold",
        color="#111827",
    )
    ax.text(
        0.35,
        7.12,
        "ColAMP-WGAN integrates AMP data, collagen sequence windows, WGAN-GP generation, predictor-guided rewards and multi-objective candidate ranking.",
        ha="left",
        va="center",
        fontsize=8.6,
        color="#4B5563",
    )

    # Layer 1: data input
    amp_db = add_box(ax, (0.40, 5.95), 1.55, 0.74, "AMP database", COLORS["data"], weight="bold")
    decoy = add_box(ax, (0.40, 4.92), 1.55, 0.74, "Non-AMP / composition decoys", COLORS["data"])
    collagen = add_box(ax, (0.40, 3.89), 1.55, 0.74, "Collagen type I sequences", COLORS["data"])

    # Layer 2: preprocessing and embedding
    preprocess = add_box(
        ax,
        (2.35, 5.20),
        1.88,
        0.88,
        "Sequence cleaning and length filtering",
        COLORS["encoding"],
        weight="bold",
    )
    windows = add_box(
        ax,
        (2.35, 3.98),
        1.88,
        0.88,
        "Collagen sliding-window extraction",
        COLORS["encoding"],
    )
    masked = add_box(
        ax,
        (4.63, 5.25),
        1.75,
        0.78,
        "Masked one-hot / soft sequence encoding",
        COLORS["encoding"],
    )
    esm = add_box(
        ax,
        (4.63, 4.15),
        1.75,
        0.78,
        "ESM-2 token embedding",
        COLORS["encoding"],
    )

    # Layer 3: predictor pre-training
    predictor_train = add_box(
        ax,
        (6.78, 4.67),
        1.62,
        0.78,
        "Predictor pre-training",
        COLORS["predictor"],
        weight="bold",
    )
    amp_pred = add_box(ax, (6.62, 3.70), 1.08, 0.65, "AMP predictor", COLORS["predictor"], fontsize=7.4)
    tox_pred = add_box(ax, (7.84, 3.70), 1.08, 0.65, "Toxicity predictor", COLORS["predictor"], fontsize=7.4)
    col_score = add_box(
        ax,
        (7.23, 2.87),
        1.08,
        0.65,
        "Collagen G-X-Y scorer",
        COLORS["predictor"],
        fontsize=7.2,
    )

    # Layer 4: adversarial generation
    z_box = add_box(ax, (6.70, 6.22), 1.10, 0.58, "Random latent vector z", "#F2F4F7", fontsize=7.5)
    generator = add_box(
        ax,
        (8.32, 6.02),
        1.58,
        0.95,
        "Generator\nMLP + sequence and length heads",
        COLORS["generator"],
        weight="bold",
        wrap_width=24,
    )
    generated = add_box(
        ax,
        (10.42, 6.12),
        1.42,
        0.74,
        "Generated peptide distributions",
        COLORS["generator"],
    )
    real = add_box(
        ax,
        (10.12, 4.86),
        1.72,
        0.78,
        "Real AMP peptides + collagen-like windows",
        COLORS["data"],
    )
    critic = add_box(
        ax,
        (12.36, 5.42),
        1.48,
        0.95,
        "Discriminator / Critic\nWGAN-GP",
        COLORS["critic"],
        weight="bold",
        wrap_width=18,
    )
    objective = add_box(
        ax,
        (9.04, 3.06),
        2.58,
        1.04,
        "Differentiable generator objectives: AMP reward, toxicity penalty, collagen motif reward, diversity, length and composition regularization",
        "#F7F3FF",
        edgecolor="#A78BFA",
        fontsize=7.05,
        wrap_width=39,
    )

    # Layer 5: screening
    library = add_box(
        ax,
        (12.48, 3.64),
        1.42,
        0.70,
        "Generated peptide library",
        COLORS["screening"],
    )
    screening = add_box(
        ax,
        (14.34, 4.52),
        1.65,
        1.02,
        "Multi-objective scoring and filtering",
        COLORS["screening"],
        weight="bold",
    )
    rules = [
        ("AMP probability", 14.08, 3.67),
        ("Low toxicity", 15.08, 3.67),
        ("Collagen motif", 14.08, 3.05),
        ("Physicochemical rules", 15.08, 3.05),
        ("Novelty / diversity", 14.58, 2.43),
    ]
    for label, x, y in rules:
        add_box(
            ax,
            (x, y),
            0.92,
            0.42,
            label,
            "#FFF7D6",
            edgecolor="#D6B75E",
            fontsize=6.35,
            radius=0.045,
            linewidth=0.65,
            wrap_width=14,
        )

    # Layer 6: final output
    candidates = add_box(
        ax,
        (16.36, 4.80),
        0.78,
        0.92,
        "Optimized collagen-derived AMP candidates",
        COLORS["final"],
        edgecolor=COLORS["final"],
        textcolor="white",
        fontsize=7.0,
        weight="bold",
        wrap_width=13,
    )
    validation = add_box(
        ax,
        (16.36, 3.36),
        0.78,
        0.88,
        "Synthesis and biological validation",
        COLORS["final2"],
        edgecolor=COLORS["final2"],
        textcolor="white",
        fontsize=7.0,
        weight="bold",
        wrap_width=13,
    )

    # Main arrows
    add_arrow(ax, (1.95, 6.32), (2.35, 5.80))
    add_arrow(ax, (1.95, 5.29), (2.35, 5.58))
    add_arrow(ax, (1.95, 4.26), (2.35, 4.42))
    add_arrow(ax, (4.23, 5.64), (4.63, 5.64))
    add_arrow(ax, (4.23, 4.42), (4.63, 4.54))
    add_arrow(ax, (6.38, 4.54), (6.78, 5.02))
    add_arrow(ax, (6.38, 5.64), (6.78, 5.21))

    add_arrow(ax, (7.80, 6.51), (8.32, 6.51))
    add_arrow(ax, (9.90, 6.51), (10.42, 6.51))
    add_arrow(ax, (11.84, 6.49), (12.36, 6.02))
    add_arrow(ax, (11.84, 5.25), (12.36, 5.80))
    add_arrow(
        ax,
        (12.36, 6.30),
        (9.90, 6.86),
        color="#7C3AED",
        lw=1.0,
        style="--",
        connectionstyle="arc3,rad=0.30",
    )
    ax.text(
        11.10,
        7.08,
        "adversarial feedback",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#5B21B6",
    )
    add_arrow(ax, (9.55, 6.02), (10.10, 4.10), color="#7C3AED", lw=0.95, style=":")
    add_arrow(ax, (7.16, 3.70), (9.04, 3.74), color="#4B8A4E", lw=0.95, style=":")
    add_arrow(ax, (8.38, 3.70), (9.04, 3.56), color="#4B8A4E", lw=0.95, style=":")
    add_arrow(ax, (7.77, 2.87), (9.04, 3.32), color="#4B8A4E", lw=0.95, style=":")
    add_arrow(ax, (11.62, 3.58), (12.48, 3.98))
    add_arrow(ax, (13.90, 3.98), (14.34, 4.88))
    add_arrow(ax, (15.99, 5.03), (16.36, 5.26))
    add_arrow(ax, (16.75, 4.80), (16.75, 4.24), color="#0F4C3A")

    # Small annotations
    add_label(ax, 12.95, 6.58, "real/fake score")
    add_label(ax, 11.00, 6.96, "temperature + length sampling")
    add_label(ax, 12.85, 5.03, "gradient penalty")
    ax.text(
        9.96,
        2.78,
        "Collagen-likeness is implemented as a G-X-Y motif/frame scorer rather than an independently trained neural classifier.",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#6B7280",
        style="italic",
    )

    # Layer guide
    bands = [
        (0.40, 1.55, "Layer 1\nData input"),
        (2.35, 4.03, "Layer 2\nPreprocessing and embedding"),
        (6.62, 2.30, "Layer 3\nPredictor pre-training"),
        (8.32, 5.52, "Layer 4\nAdversarial generation"),
        (14.08, 1.91, "Layer 5\nScreening"),
        (16.36, 0.78, "Layer 6\nCandidate output"),
    ]
    for x, width, title in bands:
        add_layer_band(ax, x, width, title)

    legend_items = [
        ("Data input", COLORS["data"]),
        ("Encoding", COLORS["encoding"]),
        ("Generator", COLORS["generator"]),
        ("Critic", COLORS["critic"]),
        ("Predictor/scorer", COLORS["predictor"]),
        ("Filtering", COLORS["screening"]),
    ]
    lx = 0.40
    for label, color in legend_items:
        add_box(
            ax,
            (lx, 0.88),
            0.34,
            0.18,
            "",
            color,
            edgecolor="#94A3B8",
            radius=0.025,
            linewidth=0.5,
        )
        ax.text(lx + 0.42, 0.97, label, ha="left", va="center", fontsize=7.2, color="#4B5563")
        lx += 1.58

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for suffix in ("svg", "pdf", "png"):
        output = OUT_DIR / f"colamp_wgan_natcomm_architecture.{suffix}"
        fig.savefig(output, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
