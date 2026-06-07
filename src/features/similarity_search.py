from __future__ import annotations


def identity_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    matches = sum(x == y for x, y in zip(a[:length], b[:length]))
    return matches / max(len(a), len(b))


def max_similarity(sequence: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(identity_similarity(sequence, ref) for ref in references)
