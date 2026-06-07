from __future__ import annotations

from dataclasses import asdict, dataclass

from src.utils.sequence_utils import clean_sequence


@dataclass(frozen=True)
class CollagenStructureFeatures:
    sequence: str
    length: int
    collagen_motif_score: float
    collagen_gxy_frame: int
    glycine_frame_fraction: float
    gxy_triplet_fraction: float
    xy_proline_fraction: float
    glycine_fraction: float
    proline_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _frame_features(sequence: str, frame: int) -> tuple[float, float, float]:
    gly_positions = list(range(frame, len(sequence), 3))
    glycine_frame_fraction = _safe_mean([1.0 if sequence[pos] == "G" else 0.0 for pos in gly_positions])

    triplet_starts = list(range(frame, max(frame, len(sequence) - 2), 3))
    gxy_triplet_fraction = _safe_mean([1.0 if sequence[pos] == "G" else 0.0 for pos in triplet_starts])

    xy_positions: list[int] = []
    for pos in triplet_starts:
        if pos + 1 < len(sequence):
            xy_positions.append(pos + 1)
        if pos + 2 < len(sequence):
            xy_positions.append(pos + 2)
    xy_proline_fraction = _safe_mean([1.0 if sequence[pos] == "P" else 0.0 for pos in xy_positions])
    return glycine_frame_fraction, gxy_triplet_fraction, xy_proline_fraction


def collagen_structure_features(sequence: str) -> CollagenStructureFeatures:
    """Return frame-invariant features for collagen-like G-X-Y preservation."""
    cleaned = clean_sequence(sequence)
    if not cleaned:
        return CollagenStructureFeatures(
            sequence="",
            length=0,
            collagen_motif_score=0.0,
            collagen_gxy_frame=0,
            glycine_frame_fraction=0.0,
            gxy_triplet_fraction=0.0,
            xy_proline_fraction=0.0,
            glycine_fraction=0.0,
            proline_fraction=0.0,
        )

    frame_rows = []
    for frame in range(3):
        glycine_frame_fraction, gxy_triplet_fraction, xy_proline_fraction = _frame_features(cleaned, frame)
        score = (
            0.65 * glycine_frame_fraction
            + 0.20 * xy_proline_fraction
            + 0.15 * gxy_triplet_fraction
        )
        frame_rows.append((score, frame, glycine_frame_fraction, gxy_triplet_fraction, xy_proline_fraction))

    score, frame, glycine_frame_fraction, gxy_triplet_fraction, xy_proline_fraction = max(
        frame_rows,
        key=lambda row: row[0],
    )
    return CollagenStructureFeatures(
        sequence=cleaned,
        length=len(cleaned),
        collagen_motif_score=float(min(max(score, 0.0), 1.0)),
        collagen_gxy_frame=int(frame),
        glycine_frame_fraction=float(glycine_frame_fraction),
        gxy_triplet_fraction=float(gxy_triplet_fraction),
        xy_proline_fraction=float(xy_proline_fraction),
        glycine_fraction=cleaned.count("G") / len(cleaned),
        proline_fraction=cleaned.count("P") / len(cleaned),
    )


def collagen_motif_score(sequence: str) -> float:
    """Return the best-frame collagen-like G-X-Y motif score."""
    return collagen_structure_features(sequence).collagen_motif_score
