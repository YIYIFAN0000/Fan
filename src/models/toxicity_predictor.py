from __future__ import annotations

from src.models.amp_predictor import SequencePredictor


class ToxicityPredictor(SequencePredictor):
    """Same CNN backbone, trained with heuristic toxicity labels."""
