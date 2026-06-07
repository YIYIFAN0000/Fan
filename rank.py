from __future__ import annotations

import argparse

from src.utils.runtime import configure_runtime

configure_runtime()

from src.inference.filter_candidates import filter_candidates
from src.inference.rank_candidates import rank_candidates
from src.utils.metrics import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    parser.add_argument("--input", default=None, help="Optional generated peptide CSV.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    filtered = filter_candidates(config, args.input)
    rank_candidates(config)


if __name__ == "__main__":
    main()
