from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.runtime import configure_runtime

configure_runtime()

import pandas as pd

from src.utils.metrics import load_yaml
from src.utils.sequence_utils import clean_sequence, is_valid_peptide


AMPlify_NON_AMP_URLS = {
    "AMPlify_non_AMP_train_balanced.fa": "https://raw.githubusercontent.com/bcgsc/AMPlify/master/data/AMPlify_non_AMP_train_balanced.fa",
    "AMPlify_non_AMP_test_balanced.fa": "https://raw.githubusercontent.com/bcgsc/AMPlify/master/data/AMPlify_non_AMP_test_balanced.fa",
    "AMPlify_non_AMP_train_imbalanced.fa": "https://raw.githubusercontent.com/bcgsc/AMPlify/master/data/AMPlify_non_AMP_train_imbalanced.fa",
    "AMPlify_non_AMP_test_imbalanced.fa": "https://raw.githubusercontent.com/bcgsc/AMPlify/master/data/AMPlify_non_AMP_test_imbalanced.fa",
}


def parse_fasta(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        sequence = clean_sequence("".join(sequence_parts))
        if sequence:
            records.append({"record_id": header.split()[0], "source_header": header, "sequence": sequence})
        header = None
        sequence_parts = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            else:
                sequence_parts.append(line)
    flush()
    return records


def download_sources(raw_dir: Path, force: bool = False) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, url in AMPlify_NON_AMP_URLS.items():
        output = raw_dir / filename
        if force or not output.exists():
            with urlopen(url, timeout=120) as response:
                output.write_bytes(response.read())
        paths.append(output)
    return paths


def load_excluded_sequences(config: dict) -> set[str]:
    excluded: set[str] = set()
    for key in ("amp_positive_csv", "collagen_windows_csv", "train_dataset_csv"):
        path = Path(config["paths"].get(key, ""))
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["sequence"])
        except Exception:
            continue
        excluded.update(df["sequence"].astype(str).map(clean_sequence).dropna().tolist())
    return {sequence for sequence in excluded if sequence}


def positive_length_distribution(config: dict) -> list[int]:
    min_len = int(config["data"]["min_len"])
    max_len = int(config["data"]["max_len"])
    path = Path(config["paths"]["amp_positive_csv"])
    if not path.exists():
        return list(range(min_len, max_len + 1))
    df = pd.read_csv(path, usecols=["sequence"])
    lengths = [
        len(sequence)
        for sequence in df["sequence"].astype(str).map(clean_sequence)
        if is_valid_peptide(sequence, min_len, max_len)
    ]
    return lengths or list(range(min_len, max_len + 1))


def collect_negatives(config: dict, force_download: bool = False) -> pd.DataFrame:
    data_cfg = config["data"]
    min_len = int(data_cfg["min_len"])
    max_len = int(data_cfg["max_len"])
    target_n = int(data_cfg["negative_samples"])
    seed = int(config["seed"])
    rng = random.Random(seed)

    raw_dir = Path(config["paths"]["raw_dir"]) / "non_amp_amplify"
    source_paths = download_sources(raw_dir, force=force_download)
    excluded = load_excluded_sequences(config)
    target_lengths = positive_length_distribution(config)

    whole_rows: list[dict] = []
    long_records: list[dict] = []
    for source_path in source_paths:
        for record in parse_fasta(source_path):
            sequence = record["sequence"]
            base = {
                "source_file": source_path.name,
                "source_class": "real_non_amp_amplify",
                "record_id": record["record_id"],
                "source_header": record["source_header"],
                "original_length": len(sequence),
            }
            if is_valid_peptide(sequence, min_len, max_len):
                if sequence not in excluded:
                    whole_rows.append({**base, "sequence": sequence, "window_start": 0, "windowed": False})
            elif len(sequence) > max_len:
                long_records.append({**base, "sequence": sequence})

    rows_by_sequence: dict[str, dict] = {row["sequence"]: row for row in whole_rows}

    attempts = 0
    max_attempts = max(1000, target_n * 200)
    while len(rows_by_sequence) < target_n and long_records and attempts < max_attempts:
        attempts += 1
        record = rng.choice(long_records)
        original = str(record["sequence"])
        allowed_lengths = [length for length in target_lengths if min_len <= length <= min(max_len, len(original))]
        length = rng.choice(allowed_lengths or list(range(min_len, min(max_len, len(original)) + 1)))
        start = rng.randint(0, len(original) - length)
        window = original[start : start + length]
        if window in rows_by_sequence or window in excluded:
            continue
        if not is_valid_peptide(window, min_len, max_len):
            continue
        rows_by_sequence[window] = {
            "sequence": window,
            "source_file": record["source_file"],
            "source_class": "real_non_amp_amplify_window",
            "record_id": record["record_id"],
            "source_header": record["source_header"],
            "original_length": record["original_length"],
            "window_start": start,
            "windowed": True,
        }

    rows = list(rows_by_sequence.values())
    rng.shuffle(rows)
    rows = rows[:target_n]
    df = pd.DataFrame(rows)
    df["label"] = 0
    ordered_columns = [
        "sequence",
        "source_file",
        "source_class",
        "label",
        "record_id",
        "source_header",
        "original_length",
        "window_start",
        "windowed",
    ]
    return df[ordered_columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    df = collect_negatives(config, force_download=args.force_download)
    output = Path(config["paths"]["non_amp_negative_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} real-source non-AMP negatives to {output}")
    print(df["source_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
