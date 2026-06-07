from __future__ import annotations

from src.utils.runtime import configure_runtime

configure_runtime()

from src.inference.generate_peptides import main


if __name__ == "__main__":
    main()
