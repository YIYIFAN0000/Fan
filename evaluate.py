from __future__ import annotations

import argparse

from src.utils.runtime import configure_runtime

configure_runtime()

from src.evaluation.figures import write_all_figures
from src.utils.metrics import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colamp_gan.yaml")
    args = parser.parse_args()
    figures = write_all_figures(load_yaml(args.config))
    if figures:
        print("Saved figures:")
        for path in figures:
            print(path)
    else:
        print("No figure inputs were found. Run training, generation, and ranking first.")


if __name__ == "__main__":
    main()
