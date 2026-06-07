from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
INDEX_TO_AA = {i: aa for aa, i in AA_TO_INDEX.items()}

HYDROPHOBIC = set("AILMFWVY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")

AA_MASS = {
    "A": 89.09,
    "C": 121.16,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
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
    passed_filter: bool


def clean_sequence(sequence: str) -> str:
    return "".join(aa for aa in sequence.upper() if aa in AA_TO_INDEX)


def net_charge(sequence: str) -> float:
    return float(sum(aa in POSITIVE for aa in sequence) - sum(aa in NEGATIVE for aa in sequence))


def hydrophobic_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(aa in HYDROPHOBIC for aa in sequence) / len(sequence)


def molecular_weight(sequence: str) -> float:
    if not sequence:
        return 0.0
    water_loss = 18.015 * max(len(sequence) - 1, 0)
    return sum(AA_MASS[aa] for aa in sequence) - water_loss


def max_repeat_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    return max(counts.values()) / len(sequence)


def describe_peptide(sequence: str, filter_config: dict) -> PeptideProperties:
    sequence = clean_sequence(sequence)
    charge = net_charge(sequence)
    hydro = hydrophobic_fraction(sequence)
    cys = sequence.count("C")
    repeat = max_repeat_fraction(sequence)
    passed = (
        charge >= float(filter_config["min_charge"])
        and hydro >= float(filter_config["min_hydrophobic_fraction"])
        and hydro <= float(filter_config["max_hydrophobic_fraction"])
        and cys <= int(filter_config["max_cysteines"])
        and repeat <= float(filter_config["max_repeat_fraction"])
    )
    return PeptideProperties(
        sequence=sequence,
        length=len(sequence),
        net_charge=charge,
        hydrophobic_fraction=hydro,
        molecular_weight=molecular_weight(sequence),
        cysteine_count=cys,
        repeat_fraction=repeat,
        passed_filter=passed,
    )


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
