from __future__ import annotations

from pathlib import Path

from src.utils.sequence_utils import clean_sequence


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = ""
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    records.append((header, clean_sequence("".join(chunks))))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if chunks:
        records.append((header, clean_sequence("".join(chunks))))
    return [(name, seq) for name, seq in records if seq]


def write_fasta(records: list[tuple[str, str]], path: str | Path, width: int = 80) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for name, seq in records:
            handle.write(f">{name}\n")
            for start in range(0, len(seq), width):
                handle.write(seq[start : start + width] + "\n")
