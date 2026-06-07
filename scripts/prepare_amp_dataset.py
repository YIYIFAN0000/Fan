from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
SEQ_LEN = 24
STRIDE = 3
MIN_LEN = 8
MAX_LEN = 100


COLLAGEN_REPORTED = [
    {
        "source_id": "CollagenVI_GVR28",
        "sequence": "GQKGDQGPPGPMGPPGPRGAPGERGRT",
        "description": "Collagen VI alpha-chain derived AMP GVR28",
        "reference": "Vastardis et al., reported collagen VI-derived antimicrobial peptide",
    },
    {
        "source_id": "CollagenVI_SFV33",
        "sequence": "SFVARNTFKRVRNGFLMRKVAVFFSNTPTRASP",
        "description": "Collagen VI alpha-chain derived AMP SFV33",
        "reference": "Vastardis et al., reported collagen VI-derived antimicrobial peptide",
    },
    {
        "source_id": "JNP2026_REI-26",
        "sequence": "REIIPRLAVDTGLGLGRRGQKGDQGP",
        "description": "Alpha-helical collagen-encoded peptide REI-26",
        "reference": "J. Nat. Prod. 2026, alpha-helical peptides encoded in collagen",
    },
    {
        "source_id": "JNP2026_LEL-28",
        "sequence": "LELPCPREGQVGPRGPRGFPGPPGRAGD",
        "description": "Alpha-helical collagen-encoded peptide LEL-28",
        "reference": "J. Nat. Prod. 2026, alpha-helical peptides encoded in collagen",
    },
    {
        "source_id": "JNP2026_TRR-26",
        "sequence": "TRRAALFGFPGLKGRAGVMGFPGPKG",
        "description": "Alpha-helical collagen-encoded peptide TRR-26",
        "reference": "J. Nat. Prod. 2026, alpha-helical peptides encoded in collagen",
    },
    {
        "source_id": "JNP2026_LRS-21",
        "sequence": "LRSNSRAFLKKVYFLRGFQKY",
        "description": "Alpha-helical collagen-encoded peptide LRS-21",
        "reference": "J. Nat. Prod. 2026, alpha-helical peptides encoded in collagen",
    },
    {
        "source_id": "JNP2026_SPE-18",
        "sequence": "SPELPQPPSSDLLGCLRA",
        "description": "Alpha-helical collagen-encoded peptide SPE-18",
        "reference": "J. Nat. Prod. 2026, alpha-helical peptides encoded in collagen",
    },
    {
        "source_id": "JNP2026_GFD-30",
        "sequence": "GFDGLAGIPGPPGERGDPGSDGQPGPPGPS",
        "description": "Alpha-helical collagen-encoded peptide GFD-30",
        "reference": "J. Nat. Prod. 2026, alpha-helical peptides encoded in collagen",
    },
    {
        "source_id": "JNP2026_GPE-19",
        "sequence": "GPEGLRGNRGERGPRGFRG",
        "description": "Alpha-helical collagen-encoded peptide GPE-19",
        "reference": "J. Nat. Prod. 2026, alpha-helical peptides encoded in collagen",
    },
    {
        "source_id": "JNP2026_TFK-18",
        "sequence": "SFVARNTFKRVRNGFLMRKVAVFFSNTPTRASP",
        "description": "Alpha-helical collagen-encoded peptide reported as TFK-18/SFV33 sequence",
        "reference": "J. Nat. Prod. 2026, alpha-helical peptides encoded in collagen",
    },
]


def clean_sequence(sequence: str) -> str:
    compact = "".join(sequence.split()).upper()
    if not compact or any(aa not in CANONICAL_AA for aa in compact):
        return ""
    if not (MIN_LEN <= len(compact) <= MAX_LEN):
        return ""
    return compact


def parse_fasta(path: Path, source_db: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    header: str | None = None
    current: list[str] = []

    def flush() -> None:
        nonlocal header, current
        if header is None:
            return
        sequence = clean_sequence("".join(current))
        if sequence:
            source_id = header.split()[0]
            records.append(
                {
                    "sequence": sequence,
                    "source_db": source_db,
                    "source_id": source_id,
                    "description": header,
                    "reference": "",
                    "is_collagen_reported": "0",
                }
            )
        header = None
        current = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            else:
                current.append(line)
    flush()
    return records


def merge_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    by_sequence: dict[str, dict[str, str]] = {}
    for record in records:
        seq = record["sequence"]
        if seq not in by_sequence:
            by_sequence[seq] = record.copy()
            continue
        existing = by_sequence[seq]
        for key in ("source_db", "source_id", "description", "reference"):
            values = [v for v in [existing.get(key, ""), record.get(key, "")] if v]
            existing[key] = "; ".join(dict.fromkeys("; ".join(values).split("; ")))
        existing["is_collagen_reported"] = (
            "1" if existing["is_collagen_reported"] == "1" or record["is_collagen_reported"] == "1" else "0"
        )
    return sorted(by_sequence.values(), key=lambda row: (row["source_db"], row["source_id"], row["sequence"]))


def stable_split(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        groups["collagen_reported" if record["is_collagen_reported"] == "1" else "public_amp"].append(record)

    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(SEED)
    for grouped_records in groups.values():
        shuffled = grouped_records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])

    for split_records in splits.values():
        split_records.sort(key=lambda row: row["sequence"])
    return splits


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "sequence_id",
        "sequence",
        "length",
        "source_db",
        "source_id",
        "is_collagen_reported",
        "description",
        "reference",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def wrap(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def write_fasta(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            collagen = "collagen_reported=1" if row["is_collagen_reported"] == "1" else "collagen_reported=0"
            handle.write(f">{row['sequence_id']} source={row['source_db']} {collagen} source_id={row['source_id']}\n")
            handle.write(wrap(row["sequence"]) + "\n")


def make_windows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    windows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        sequence = row["sequence"]
        for start in range(0, len(sequence) - SEQ_LEN + 1, STRIDE):
            window = sequence[start : start + SEQ_LEN]
            if window in seen:
                continue
            seen.add(window)
            windows.append(
                {
                    **row,
                    "sequence": window,
                    "length": str(len(window)),
                    "sequence_id": f"{row['sequence_id']}_w{start:03d}",
                    "description": f"{row['description']} | window_start={start}",
                }
            )
    windows.sort(key=lambda row: row["sequence"])
    return windows


def add_ids(records: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for record in records:
        digest = hashlib.sha1(record["sequence"].encode("ascii")).hexdigest()[:10]
        prefix = "COL" if record["is_collagen_reported"] == "1" else "AMP"
        output.append({**record, "sequence_id": f"{prefix}_{digest}", "length": str(len(record["sequence"]))})
    return output


def main() -> None:
    records: list[dict[str, str]] = []
    records.extend(parse_fasta(DATA / "apd6_natural_amps_2024a_raw.fasta", "APD6_natural_2024a"))
    records.extend(parse_fasta(DATA / "dramp3_antimicrobial_amps_raw.fasta", "DRAMP3_antimicrobial"))
    records.extend(parse_fasta(DATA / "dramp3_natural_amps_raw.fasta", "DRAMP3_natural"))

    for row in COLLAGEN_REPORTED:
        sequence = clean_sequence(row["sequence"])
        if not sequence:
            raise ValueError(f"Invalid collagen sequence: {row['source_id']}")
        records.append(
            {
                "sequence": sequence,
                "source_db": "reported_collagen_amp",
                "source_id": row["source_id"],
                "description": row["description"],
                "reference": row["reference"],
                "is_collagen_reported": "1",
            }
        )

    merged = add_ids(merge_records(records))
    collagen = [row for row in merged if row["is_collagen_reported"] == "1"]
    splits = stable_split(merged)

    write_csv(DATA / "amp_all_clean.csv", merged)
    write_fasta(DATA / "amp_all_clean.fasta", merged)
    write_csv(DATA / "collagen_reported_amps.csv", collagen)
    write_fasta(DATA / "collagen_reported_amps.fasta", collagen)

    summary_rows = []
    for split, rows in splits.items():
        write_csv(DATA / f"{split}.csv", rows)
        write_fasta(DATA / f"{split}.fasta", rows)
        windows = make_windows(rows)
        write_csv(DATA / f"{split}_windows_{SEQ_LEN}.csv", windows)
        write_fasta(DATA / f"{split}_windows_{SEQ_LEN}.fasta", windows)
        summary_rows.append((split, len(rows), len(windows), sum(row["is_collagen_reported"] == "1" for row in rows)))

    with (DATA / "DATASET_CARD.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Antimicrobial Peptide Dataset\n\n")
        handle.write("This folder replaces the earlier sheep collagen example with curated public AMP data plus reported collagen-derived antimicrobial peptides.\n\n")
        handle.write("## Sources\n\n")
        handle.write("- APD6 natural antimicrobial peptide FASTA: `apd6_natural_amps_2024a_raw.fasta`.\n")
        handle.write("- DRAMP 3.0 antimicrobial and natural FASTA downloads: `dramp3_antimicrobial_amps_raw.fasta`, `dramp3_natural_amps_raw.fasta`.\n")
        handle.write("- Reported collagen-derived AMP sequences: `collagen_reported_amps.csv` and `.fasta`.\n\n")
        handle.write("## Cleaning\n\n")
        handle.write("- Kept canonical amino-acid sequences only: ACDEFGHIKLMNPQRSTVWY.\n")
        handle.write(f"- Kept sequence lengths from {MIN_LEN} to {MAX_LEN} amino acids.\n")
        handle.write("- Deduplicated by exact peptide sequence across all sources.\n")
        handle.write("- Split at peptide-sequence level with seed 42: 80% train, 10% validation, 10% test.\n")
        handle.write(f"- Window files use seq_len={SEQ_LEN}, stride={STRIDE}, created after splitting to avoid parent-sequence leakage.\n\n")
        handle.write("## Counts\n\n")
        handle.write("| split | peptides | 24-aa windows | collagen-reported peptides |\n")
        handle.write("|---|---:|---:|---:|\n")
        for split, n_rows, n_windows, n_collagen in summary_rows:
            handle.write(f"| {split} | {n_rows} | {n_windows} | {n_collagen} |\n")
        handle.write(f"| all | {len(merged)} | {sum(len(make_windows(rows)) for rows in splits.values())} | {len(collagen)} |\n")


if __name__ == "__main__":
    main()
