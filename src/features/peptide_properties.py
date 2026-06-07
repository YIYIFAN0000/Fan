from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

from src.utils.sequence_utils import repeat_fraction

HYDROPHOBIC = set("AILMFWVY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("STNQ")
AA_MASS = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}


@dataclass(frozen=True)
class PeptideProperties:
    sequence: str
    length: int
    net_charge: float
    hydrophobic_fraction: float
    molecular_weight: float
    cysteine_count: int
    repeat_fraction: float
    aromatic_fraction: float
    proline_fraction: float
    glycine_fraction: float
    arginine_fraction: float
    lysine_fraction: float
    positive_fraction: float
    negative_fraction: float
    polar_fraction: float
    charge_density: float
    sequence_entropy: float
    longest_run_fraction: float
    dipeptide_repeat_fraction: float
    tripeptide_repeat_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def sequence_entropy(sequence: str) -> float:
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    raw = -sum((count / len(sequence)) * math.log(count / len(sequence), 2) for count in counts.values())
    return raw / math.log(20, 2)


def longest_run_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    longest = 1
    current = 1
    for previous, aa in zip(sequence, sequence[1:]):
        if aa == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest / len(sequence)


def kmer_repeat_fraction(sequence: str, k: int) -> float:
    if len(sequence) < k:
        return 0.0
    kmers = [sequence[index : index + k] for index in range(len(sequence) - k + 1)]
    return max(Counter(kmers).values()) / len(kmers)


def compute_properties(sequence: str) -> PeptideProperties:
    length = len(sequence)
    if length == 0:
        raise ValueError("Cannot compute properties for an empty sequence.")
    mass = sum(AA_MASS[aa] for aa in sequence) - 18.015 * max(length - 1, 0)
    net_charge = float(sum(aa in POSITIVE for aa in sequence) - sum(aa in NEGATIVE for aa in sequence))
    return PeptideProperties(
        sequence=sequence,
        length=length,
        net_charge=net_charge,
        hydrophobic_fraction=sum(aa in HYDROPHOBIC for aa in sequence) / length,
        molecular_weight=mass,
        cysteine_count=sequence.count("C"),
        repeat_fraction=repeat_fraction(sequence),
        aromatic_fraction=sum(aa in "FWY" for aa in sequence) / length,
        proline_fraction=sequence.count("P") / length,
        glycine_fraction=sequence.count("G") / length,
        arginine_fraction=sequence.count("R") / length,
        lysine_fraction=sequence.count("K") / length,
        positive_fraction=sum(aa in POSITIVE for aa in sequence) / length,
        negative_fraction=sum(aa in NEGATIVE for aa in sequence) / length,
        polar_fraction=sum(aa in POLAR for aa in sequence) / length,
        charge_density=net_charge / length,
        sequence_entropy=sequence_entropy(sequence),
        longest_run_fraction=longest_run_fraction(sequence),
        dipeptide_repeat_fraction=kmer_repeat_fraction(sequence, 2),
        tripeptide_repeat_fraction=kmer_repeat_fraction(sequence, 3),
    )


def passes_rule_filter(sequence: str, cfg: dict) -> bool:
    props = compute_properties(sequence)
    return (
        props.net_charge >= float(cfg["min_charge"])
        and props.net_charge <= float(cfg.get("max_charge", props.net_charge))
        and props.charge_density <= float(cfg.get("max_charge_density", props.charge_density))
        and props.hydrophobic_fraction >= float(cfg["min_hydrophobic_fraction"])
        and props.hydrophobic_fraction <= float(cfg["max_hydrophobic_fraction"])
        and props.cysteine_count <= int(cfg["max_cysteines"])
        and props.repeat_fraction <= float(cfg["max_repeat_fraction"])
        and props.positive_fraction <= float(cfg.get("max_positive_fraction", props.positive_fraction))
        and props.arginine_fraction <= float(cfg.get("max_arginine_fraction", props.arginine_fraction))
        and props.lysine_fraction <= float(cfg.get("max_lysine_fraction", props.lysine_fraction))
        and props.proline_fraction <= float(cfg.get("max_proline_fraction", props.proline_fraction))
        and props.glycine_fraction <= float(cfg.get("max_glycine_fraction", props.glycine_fraction))
        and props.longest_run_fraction <= float(cfg.get("max_longest_run_fraction", props.longest_run_fraction))
        and props.dipeptide_repeat_fraction <= float(
            cfg.get("max_dipeptide_repeat_fraction", props.dipeptide_repeat_fraction)
        )
        and props.tripeptide_repeat_fraction <= float(
            cfg.get("max_tripeptide_repeat_fraction", props.tripeptide_repeat_fraction)
        )
        and props.sequence_entropy >= float(cfg.get("min_sequence_entropy", props.sequence_entropy))
    )
