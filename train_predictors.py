from __future__ import annotations

from src.utils.runtime import configure_runtime

configure_runtime()

from src.training.train_predictors import main


if __name__ == "__main__":
    main()
