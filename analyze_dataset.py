from __future__ import annotations

import argparse

from src.utils.runtime import configure_runtime

configure_runtime()

from src.evaluation.dataset_analysis import analyze_dataset
from src.utils.metrics import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    args = parser.parse_args()
    outputs = analyze_dataset(load_yaml(args.config))

    print("Saved dataset analysis tables:")
    for path in outputs["tables"].values():
        print(path)
    print("Saved dataset analysis figures:")
    for path in outputs["figures"]:
        print(path)


if __name__ == "__main__":
    main()
