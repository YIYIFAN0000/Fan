from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"


COLORS = {
    "ink": "#111827",
    "muted": "#5F6B7A",
    "panel_edge_blue": "#7BA7D9",
    "panel_edge_green": "#79B879",
    "panel_edge_gold": "#D4A84F",
    "panel_fill": "#FBFDFF",
    "data": "#E5EEF8",
    "data_edge": "#2E6EAE",
    "collagen": "#E6F3E2",
    "collagen_edge": "#2F7D3B",
    "model": "#EAF3FF",
    "model_edge": "#2057A8",
    "generator": "#FFF0DE",
    "generator_edge": "#D86B18",
    "critic": "#EAF2FF",
    "critic_edge": "#1E5BB8",
    "loss": "#FFF1F1",
    "loss_edge": "#CF3333",
    "reward": "#F4EEFF",
    "reward_edge": "#7D54C9",
    "screen": "#FFF6D8",
    "screen_edge": "#C89A2B",
    "artifact": "#E9F7EC",
    "artifact_edge": "#237C3A",
    "red": "#C51F1A",
    "blue": "#1859B7",
    "green": "#237C3A",
    "purple": "#7D54C9",
    "orange": "#D86B18",
    "line": "#222222",
}


def _wrap(text: str, width: int) -> str:
    lines: list[str] = []
    for chunk in text.split("\n"):
        lines.extend(wrap(chunk, width=width, break_long_words=False) or [""])
    return "\n".join(lines)


def rounded_box(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str,
    color: str = COLORS["ink"],
    fs: float = 8.0,
    weight: str = "normal",
    radius: float = 0.055,
    lw: float = 1.0,
    wrap_width: int = 22,
    align: str = "center",
    ls: str = "-",
) -> FancyBboxPatch:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
    )
    ax.add_patch(patch)
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.16
    ax.text(
        tx,
        y + h / 2,
        _wrap(text, wrap_width),
        ha=ha,
        va="center",
        fontsize=fs,
        color=color,
        weight=weight,
        linespacing=1.14,
    )
    return patch


def panel(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    title: str,
    *,
    ec: str,
    title_color: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.10",
        facecolor=COLORS["panel_fill"],
        edgecolor=ec,
        linewidth=1.15,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.22,
        y + h - 0.20,
        title,
        ha="left",
        va="center",
        fontsize=13.2,
        weight="bold",
        color=title_color,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    lw: float = 1.1,
    style: str = "-",
    rad: float = 0.0,
    ms: float = 10.5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            color=color,
            linewidth=lw,
            linestyle=style,
            shrinkA=3,
            shrinkB=3,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def cylinder(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str,
    color: str = COLORS["ink"],
    fs: float = 8.0,
    weight: str = "normal",
    wrap_width: int = 20,
) -> None:
    x, y = xy
    body = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
    )
    ax.add_patch(body)
    top = Ellipse((x + 0.28, y + h - 0.20), 0.42, 0.20, facecolor="#FFFFFF", edgecolor=ec, linewidth=1.0)
    mid = Ellipse((x + 0.28, y + h - 0.24), 0.42, 0.18, facecolor=fc, edgecolor=ec, linewidth=0.9)
    ax.add_patch(top)
    ax.add_patch(mid)
    ax.plot([x + 0.07, x + 0.07], [y + h - 0.24, y + 0.15], color=ec, lw=0.85)
    ax.plot([x + 0.49, x + 0.49], [y + h - 0.24, y + 0.15], color=ec, lw=0.85)
    ax.text(
        x + 0.66,
        y + h / 2,
        _wrap(text, wrap_width),
        ha="left",
        va="center",
        fontsize=fs,
        color=color,
        weight=weight,
        linespacing=1.15,
    )


def small_capsule(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    *,
    color: str,
    fc: str = "#FFFFFF",
    fs: float = 7.0,
    style: str = "-",
) -> None:
    rounded_box(
        ax,
        xy,
        w,
        h,
        text,
        fc=fc,
        ec=color,
        color=color,
        fs=fs,
        radius=0.045,
        lw=0.85,
        wrap_width=38,
        ls=style,
    )


def draw_sequence_dots(ax, x: float, y: float, color: str, n: int = 7) -> None:
    for i in range(n):
        ax.add_patch(Ellipse((x + i * 0.18, y), 0.11, 0.11, facecolor=color, edgecolor="white", lw=0.5))
    ax.text(x + n * 0.18 + 0.08, y, "...", ha="left", va="center", fontsize=8, color=COLORS["muted"])


def draw() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(17.8, 11.0), dpi=300)
    ax.set_xlim(0, 17.8)
    ax.set_ylim(0, 11.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    panel(ax, (0.22, 8.05), 17.36, 2.75, "Stage I. Predictor pre-training", ec=COLORS["panel_edge_blue"], title_color="#0D4EA6")
    panel(ax, (0.22, 2.63), 17.36, 5.20, "Stage II. Core adversarial training (WGAN-GP)", ec=COLORS["panel_edge_green"], title_color="#126326")
    panel(ax, (0.22, 0.30), 17.36, 2.05, "Stage III. Candidate generation, screening and ranking", ec=COLORS["panel_edge_gold"], title_color="#8A6200")

    # Stage I
    cylinder(
        ax,
        (0.62, 9.42),
        2.85,
        0.62,
        "AMP databases\nAMP positives, label = 1",
        fc=COLORS["data"],
        ec=COLORS["data_edge"],
        fs=8.5,
        weight="bold",
    )
    cylinder(
        ax,
        (0.62, 8.72),
        2.85,
        0.62,
        "Non-AMP or composition decoys\nlabel = 0",
        fc="#EEF3FA",
        ec="#607D9B",
        fs=8.0,
    )
    cylinder(
        ax,
        (0.62, 8.10),
        2.85,
        0.46,
        "Collagen windows\ncandidate pool and GAN real samples",
        fc=COLORS["collagen"],
        ec=COLORS["collagen_edge"],
        fs=7.0,
    )

    data_construct = rounded_box(
        ax,
        (4.05, 8.68),
        2.70,
        1.24,
        "Predictor dataset construction\n- clean sequences\n- merge AMP positives + non-AMP/decoy negatives\n- remove duplicates\n- train/val/test split\n- cache ESM-2 features",
        fc="#F7F4FF",
        ec="#7D69B1",
        fs=7.35,
        wrap_width=28,
        align="left",
    )
    tox_rule = rounded_box(
        ax,
        (4.05, 8.10),
        2.70,
        0.46,
        "toxicity_label = 1 if hydrophobic_fraction > 0.70 OR net_charge > 8 OR cysteine_count > 2",
        fc="#FFFFFF",
        ec=COLORS["loss_edge"],
        color=COLORS["red"],
        fs=6.9,
        wrap_width=47,
        ls="--",
    )
    amp_predictor = rounded_box(
        ax,
        (7.70, 9.22),
        3.85,
        0.72,
        "AMP Predictor\nSequencePredictor / ESMFeaturePredictor",
        fc=COLORS["model"],
        ec=COLORS["model_edge"],
        color="#0D3F8C",
        fs=8.3,
        weight="bold",
        wrap_width=34,
    )
    tox_predictor = rounded_box(
        ax,
        (7.70, 8.28),
        3.85,
        0.72,
        "Toxicity Predictor\nSequencePredictor / ESMFeaturePredictor",
        fc="#EAF7ED",
        ec=COLORS["collagen_edge"],
        color="#126326",
        fs=8.3,
        weight="bold",
        wrap_width=34,
    )
    rounded_box(
        ax,
        (11.92, 9.30),
        1.75,
        0.52,
        "BCEWithLogitsLoss\n+ Adam",
        fc="#FFFFFF",
        ec="#A2B9D9",
        fs=7.4,
        wrap_width=18,
    )
    rounded_box(
        ax,
        (11.92, 8.36),
        1.75,
        0.52,
        "BCEWithLogitsLoss\n+ Adam",
        fc="#FFFFFF",
        ec="#A9CFAF",
        fs=7.4,
        wrap_width=18,
    )
    cylinder(
        ax,
        (14.20, 9.30),
        2.15,
        0.52,
        "amp_predictor.pt",
        fc="#E3F0FF",
        ec=COLORS["model_edge"],
        color="#0D3F8C",
        fs=7.8,
        weight="bold",
    )
    cylinder(
        ax,
        (14.20, 8.36),
        2.15,
        0.52,
        "toxicity_predictor.pt",
        fc="#EAF7ED",
        ec=COLORS["collagen_edge"],
        color="#126326",
        fs=7.8,
        weight="bold",
    )

    arrow(ax, (3.47, 9.72), (4.05, 9.35))
    arrow(ax, (3.47, 9.02), (4.05, 9.03))
    arrow(ax, (6.75, 9.24), (7.70, 9.58))
    arrow(ax, (6.75, 8.34), (7.70, 8.64))
    arrow(ax, (11.55, 9.58), (11.92, 9.56))
    arrow(ax, (11.55, 8.64), (11.92, 8.62))
    arrow(ax, (13.67, 9.56), (14.20, 9.56))
    arrow(ax, (13.67, 8.62), (14.20, 8.62))

    ax.text(8.90, 7.98, "frozen during GAN training", ha="center", va="center", fontsize=8.2, color=COLORS["blue"], weight="bold")
    arrow(ax, (8.90, 8.05), (8.90, 7.58), color=COLORS["blue"], lw=1.5, ms=14)

    # Stage II: WGAN-GP core
    noise = rounded_box(
        ax,
        (0.62, 4.95),
        1.20,
        1.25,
        "A. Random noise\nz ~ N(0, I)",
        fc="#FBF8FF",
        ec=COLORS["reward_edge"],
        color="#3C247A",
        fs=7.9,
        weight="bold",
        wrap_width=18,
    )
    for i in range(4):
        ax.add_patch(Ellipse((1.22, 5.17 + i * 0.22), 0.18, 0.18, facecolor="#E9DDFF", edgecolor=COLORS["reward_edge"], lw=0.9))
    ax.text(1.22, 5.98, "...", ha="center", va="center", fontsize=9, color="#3C247A")

    gen_poly = Polygon(
        [(2.55, 4.36), (4.45, 4.95), (4.45, 6.55), (2.55, 7.05)],
        closed=True,
        facecolor=COLORS["generator"],
        edgecolor=COLORS["generator_edge"],
        linewidth=1.15,
    )
    ax.add_patch(gen_poly)
    ax.text(3.55, 6.30, "B. Generator (G)", ha="center", va="center", fontsize=10.5, weight="bold", color=COLORS["orange"])
    ax.text(
        3.55,
        5.82,
        "MLP backbone\nsequence head: [B, L, 20]\nlength head: [B, len_classes]",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#4B2B12",
        linespacing=1.15,
    )
    # A small network glyph
    nodes = [(3.03, 5.17), (3.35, 5.40), (3.68, 5.18), (3.36, 4.88), (4.00, 5.46), (4.06, 4.94)]
    for a, b in [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 5), (0, 3)]:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=COLORS["orange"], lw=0.8)
    for x, y in nodes:
        ax.add_patch(Ellipse((x, y), 0.13, 0.13, facecolor="#FFE0BF", edgecolor=COLORS["orange"], lw=1.0))

    fake = rounded_box(
        ax,
        (5.05, 5.18),
        1.92,
        0.96,
        "C. Fake peptide sample\nsoft probabilities + length mask",
        fc="#FFFFFF",
        ec=COLORS["generator_edge"],
        color=COLORS["orange"],
        fs=7.7,
        weight="bold",
        wrap_width=24,
    )
    draw_sequence_dots(ax, 5.42, 5.36, "#F28C28")

    real_data = rounded_box(
        ax,
        (4.85, 6.60),
        1.98,
        0.62,
        "D. Real peptide samples\nAMP + collagen-like windows",
        fc=COLORS["collagen"],
        ec=COLORS["collagen_edge"],
        color="#135E22",
        fs=7.4,
        weight="bold",
        wrap_width=28,
    )
    draw_sequence_dots(ax, 5.15, 6.72, "#5AA357", n=6)

    critic_poly = Polygon(
        [(8.82, 4.62), (10.28, 5.10), (10.28, 6.72), (8.82, 7.10)],
        closed=True,
        facecolor=COLORS["critic"],
        edgecolor=COLORS["critic_edge"],
        linewidth=1.15,
    )
    ax.add_patch(critic_poly)
    ax.text(9.58, 6.46, "E. Critic /\nDiscriminator (D)", ha="center", va="center", fontsize=10.0, weight="bold", color=COLORS["blue"], linespacing=1.0)
    ax.text(9.58, 5.82, "WGAN-GP critic\nmasked one-hot input", ha="center", va="center", fontsize=7.5, color="#1D3557")
    nodes2 = [(9.22, 5.35), (9.45, 5.55), (9.70, 5.35), (9.94, 5.55)]
    for a, b in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]:
        ax.plot([nodes2[a][0], nodes2[b][0]], [nodes2[a][1], nodes2[b][1]], color=COLORS["blue"], lw=0.7)
    for x, y in nodes2:
        ax.add_patch(Ellipse((x, y), 0.10, 0.10, facecolor="#BFD7FF", edgecolor=COLORS["blue"], lw=0.9))

    adv_loss = rounded_box(
        ax,
        (8.10, 3.92),
        2.15,
        0.66,
        "F. Adversarial loss\nE[D(fake)] - E[D(real)] + lambda * GP",
        fc=COLORS["loss"],
        ec=COLORS["loss_edge"],
        color=COLORS["red"],
        fs=7.4,
        weight="bold",
        wrap_width=31,
    )
    gp = rounded_box(
        ax,
        (6.60, 3.00),
        3.02,
        0.62,
        "Gradient penalty (GP): interpolate real/fake; autograd gradients; mean((||grad||2 - 1)^2); repeat critic_steps",
        fc="#FFFFFF",
        ec=COLORS["loss_edge"],
        color=COLORS["red"],
        fs=6.9,
        wrap_width=62,
        ls="--",
    )

    reward_panel = rounded_box(
        ax,
        (11.15, 5.62),
        3.65,
        1.90,
        "",
        fc="#FFFFFF",
        ec=COLORS["model_edge"],
        color=COLORS["blue"],
        fs=8.5,
        weight="bold",
        wrap_width=30,
    )
    ax.text(
        12.98,
        7.34,
        "Reward guidance (frozen predictors)",
        ha="center",
        va="center",
        fontsize=8.5,
        weight="bold",
        color=COLORS["blue"],
    )
    small_capsule(ax, (11.42, 6.80), 3.08, 0.32, "Frozen AMP predictor -> amp_reward", color=COLORS["blue"], fs=6.9)
    small_capsule(ax, (11.42, 6.36), 3.08, 0.32, "Frozen toxicity predictor -> tox_penalty", color=COLORS["green"], fs=6.9)
    small_capsule(ax, (11.42, 5.92), 3.08, 0.32, "collagen_soft_reward: G every 3rd position + Pro score", color=COLORS["purple"], fs=6.55)
    ax.text(
        12.98,
        5.74,
        "soft_esm_features(fake) keeps predictor rewards differentiable",
        ha="center",
        va="center",
        fontsize=6.25,
        color=COLORS["muted"],
    )

    objective = rounded_box(
        ax,
        (11.35, 4.30),
        3.25,
        1.12,
        "Generator objective (minimize)\nL_G = -E[D(fake)] - w_amp * amp_reward + w_tox * tox_penalty - w_col * collagen_reward - w_div * diversity + w_len * length_KL + regularizers",
        fc=COLORS["loss"],
        ec=COLORS["loss_edge"],
        color=COLORS["red"],
        fs=7.05,
        weight="bold",
        wrap_width=48,
    )
    trained = rounded_box(
        ax,
        (11.35, 3.08),
        3.25,
        0.78,
        "Trained peptide generator\nGoal: collagen-like AMP candidates with high AMP score and low predicted toxicity",
        fc=COLORS["artifact"],
        ec=COLORS["artifact_edge"],
        color="#0C5724",
        fs=7.6,
        weight="bold",
        wrap_width=42,
    )
    cylinder(
        ax,
        (14.92, 3.18),
        1.92,
        0.58,
        "generator.pt",
        fc=COLORS["artifact"],
        ec=COLORS["artifact_edge"],
        color="#0C5724",
        fs=7.6,
        weight="bold",
    )

    # Stage II arrows
    arrow(ax, (1.82, 5.58), (2.55, 5.58))
    arrow(ax, (4.45, 5.60), (5.05, 5.60))
    arrow(ax, (6.97, 5.65), (8.82, 5.58))
    arrow(ax, (6.83, 6.91), (8.82, 6.42))
    arrow(ax, (10.28, 5.54), (11.35, 4.92), rad=-0.10)
    arrow(ax, (10.00, 4.62), (9.55, 4.58), color=COLORS["red"])
    arrow(ax, (9.10, 3.92), (8.95, 3.62), color=COLORS["red"])
    arrow(ax, (9.50, 3.62), (9.75, 3.92), color=COLORS["red"])
    arrow(ax, (5.98, 5.18), (8.10, 4.26), color=COLORS["red"], rad=0.05)
    arrow(ax, (11.15, 6.55), (10.28, 6.10), color=COLORS["blue"], style=":", rad=0.0)
    arrow(ax, (13.00, 5.80), (13.00, 5.42), color=COLORS["purple"])
    arrow(ax, (13.00, 4.30), (13.00, 3.86), color=COLORS["green"])
    arrow(ax, (14.60, 3.47), (14.92, 3.47), color=COLORS["green"])
    arrow(ax, (11.35, 3.47), (10.02, 3.92), color=COLORS["red"], style="--", rad=-0.20)
    arrow(ax, (8.10, 4.02), (3.55, 4.36), color=COLORS["red"], lw=1.35, rad=0.18, ms=13)
    ax.text(4.80, 3.78, "G. adversarial feedback", ha="center", va="center", fontsize=9.2, weight="bold", color=COLORS["red"])

    # Stage III
    z2 = rounded_box(
        ax,
        (0.62, 1.12),
        1.15,
        0.55,
        "z sampling",
        fc="#FBF8FF",
        ec=COLORS["reward_edge"],
        color="#3C247A",
        fs=7.3,
        weight="bold",
    )
    gen2 = rounded_box(
        ax,
        (2.18, 0.98),
        1.80,
        0.78,
        "Trained Generator\ntemperature + length sampling",
        fc=COLORS["generator"],
        ec=COLORS["generator_edge"],
        color=COLORS["orange"],
        fs=7.1,
        weight="bold",
        wrap_width=25,
    )
    cylinder(
        ax,
        (4.40, 1.10),
        2.16,
        0.56,
        "generated_peptides.csv",
        fc="#FFF8E8",
        ec=COLORS["screen_edge"],
        color="#8A6200",
        fs=7.4,
        weight="bold",
    )
    filter_box = rounded_box(
        ax,
        (6.94, 0.76),
        3.45,
        1.12,
        "Multi-objective filtering\noutput: filtered_candidates.csv\nAMP probability; toxicity probability; collagen G-X-Y score; charge/hydrophobicity; complexity and repeat rules; similarity to collagen windows",
        fc=COLORS["screen"],
        ec=COLORS["screen_edge"],
        color="#7A5200",
        fs=6.95,
        weight="bold",
        wrap_width=56,
    )
    rank_box = rounded_box(
        ax,
        (10.80, 0.88),
        2.30,
        0.88,
        "Weighted ranking + diverse top-k selection",
        fc="#FFF9E8",
        ec=COLORS["screen_edge"],
        color="#7A5200",
        fs=7.6,
        weight="bold",
        wrap_width=28,
    )
    cylinder(
        ax,
        (13.55, 1.18),
        2.06,
        0.52,
        "ranked_candidates_top100.csv",
        fc=COLORS["artifact"],
        ec=COLORS["artifact_edge"],
        color="#0C5724",
        fs=6.8,
        weight="bold",
        wrap_width=20,
    )
    cylinder(
        ax,
        (13.55, 0.58),
        2.06,
        0.52,
        "synthesis_shortlist_50.csv",
        fc=COLORS["artifact"],
        ec=COLORS["artifact_edge"],
        color="#0C5724",
        fs=6.8,
        weight="bold",
        wrap_width=20,
    )
    validation = rounded_box(
        ax,
        (16.02, 0.80),
        1.24,
        0.98,
        "Synthesis and biological validation",
        fc="#0F4C3A",
        ec="#0F4C3A",
        color="white",
        fs=7.4,
        weight="bold",
        wrap_width=17,
    )

    arrow(ax, (1.77, 1.40), (2.18, 1.40))
    arrow(ax, (3.98, 1.40), (4.40, 1.40))
    arrow(ax, (6.56, 1.40), (6.94, 1.40))
    arrow(ax, (10.39, 1.32), (10.80, 1.32))
    arrow(ax, (13.10, 1.32), (13.55, 1.42))
    arrow(ax, (13.10, 1.10), (13.55, 0.84))
    arrow(ax, (15.61, 0.84), (16.02, 1.10), color=COLORS["green"])
    arrow(ax, (15.61, 1.42), (16.02, 1.42), color=COLORS["green"])

    # Cross-stage use arrows
    arrow(ax, (14.20, 9.30), (13.00, 7.52), color=COLORS["blue"], style="--", rad=-0.18)
    arrow(ax, (14.20, 8.36), (13.00, 7.08), color=COLORS["green"], style="--", rad=-0.08)
    # Legend
    legend_y = 0.05
    legend_items = [
        ("Data / input", COLORS["data"], COLORS["data_edge"]),
        ("Model / component", COLORS["model"], COLORS["model_edge"]),
        ("Generator / sampling", COLORS["generator"], COLORS["generator_edge"]),
        ("Critic", COLORS["critic"], COLORS["critic_edge"]),
        ("Loss / objective", COLORS["loss"], COLORS["loss_edge"]),
        ("Reward / signal", COLORS["reward"], COLORS["reward_edge"]),
        ("Screening / output", COLORS["screen"], COLORS["screen_edge"]),
    ]
    x = 0.55
    for label, fc, ec in legend_items:
        rounded_box(ax, (x, legend_y), 0.27, 0.15, "", fc=fc, ec=ec, radius=0.02, lw=0.65)
        ax.text(x + 0.34, legend_y + 0.075, label, ha="left", va="center", fontsize=6.5, color=COLORS["muted"])
        x += 2.25
    ax.plot([14.85, 15.25], [legend_y + 0.075, legend_y + 0.075], color=COLORS["red"], lw=1.4)
    ax.add_patch(FancyArrowPatch((15.05, legend_y + 0.075), (15.25, legend_y + 0.075), arrowstyle="-|>", mutation_scale=8, color=COLORS["red"], lw=1.4))
    ax.text(15.35, legend_y + 0.075, "adversarial feedback", ha="left", va="center", fontsize=6.5, color=COLORS["muted"])

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            OUT_DIR / f"colamp_wgan_top_journal_architecture.{suffix}",
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    draw()
