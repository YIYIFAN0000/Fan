from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import torch

from src.utils.runtime import configure_runtime
from src.utils.sequence_utils import AMINO_ACIDS, clean_sequence

configure_runtime()


def esm_config(config: dict) -> dict:
    cfg = dict(config.get("esm", {}))
    cfg.setdefault("enabled", False)
    cfg.setdefault("model_name", "facebook/esm2_t6_8M_UR50D")
    cfg.setdefault("local_files_only", True)
    cfg.setdefault("embedding_dim", 320)
    cfg.setdefault("embedding_mode", "token")
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("cache_dir", "data/processed/esm2_cache")
    return cfg


def use_esm_features(config: dict) -> bool:
    return bool(esm_config(config).get("enabled", False))


def esm_feature_dim(config: dict) -> int:
    return int(esm_config(config)["embedding_dim"])


def _sequence_fingerprint(sequences: list[str], cfg: dict) -> str:
    payload = "\n".join(
        [
            str(cfg["model_name"]),
            str(cfg["embedding_mode"]),
            *sequences,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(config: dict, sequences: list[str], cache_name: str | None) -> Path:
    cfg = esm_config(config)
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = cache_name or "esm2"
    return cache_dir / f"{prefix}_{_sequence_fingerprint(sequences, cfg)}.pt"


def _load_tokenizer_and_model(config: dict, device: torch.device):
    cfg = esm_config(config)
    try:
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as transformers_logging
    except ImportError as exc:
        raise ImportError(
            "ESM-2 features require transformers. Install it with `pip install transformers`."
        ) from exc
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        local_files_only=bool(cfg["local_files_only"]),
    )
    model = AutoModel.from_pretrained(
        cfg["model_name"],
        local_files_only=bool(cfg["local_files_only"]),
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model


def _amino_acid_token_ids(tokenizer) -> torch.Tensor:
    token_ids: list[int] = []
    for aa in AMINO_ACIDS:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id is None or token_id == tokenizer.unk_token_id:
            encoded = tokenizer(aa, add_special_tokens=False)
            ids = encoded.get("input_ids", [])
            if not ids:
                raise ValueError(f"ESM tokenizer cannot encode amino acid token {aa!r}.")
            token_id = int(ids[0])
        token_ids.append(int(token_id))
    return torch.tensor(token_ids, dtype=torch.long)


def load_esm_amino_acid_embeddings(config: dict, device: torch.device) -> torch.Tensor:
    tokenizer, model = _load_tokenizer_and_model(config, device)
    token_ids = _amino_acid_token_ids(tokenizer).to(device)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(token_ids)
    return embeddings.detach()


def soft_esm_features(
    probabilities: torch.Tensor,
    amino_acid_embeddings: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project soft amino-acid probabilities into differentiable ESM-2 token features."""
    token_features = probabilities @ amino_acid_embeddings
    if mask is not None:
        weighted = token_features * mask.unsqueeze(-1)
        return weighted.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return token_features.mean(dim=1)


def _token_embedding_features(sequences: list[str], config: dict, device: torch.device) -> torch.Tensor:
    tokenizer, model = _load_tokenizer_and_model(config, device)
    token_ids = _amino_acid_token_ids(tokenizer).to(device)
    with torch.no_grad():
        aa_embeddings = model.get_input_embeddings()(token_ids)
    rows: list[torch.Tensor] = []
    aa_to_idx = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
    for sequence in sequences:
        indices = torch.tensor([aa_to_idx[aa] for aa in sequence], dtype=torch.long, device=device)
        rows.append(aa_embeddings.index_select(0, indices).mean(dim=0))
    return torch.stack(rows).cpu()


def _contextual_features(sequences: list[str], config: dict, device: torch.device) -> torch.Tensor:
    cfg = esm_config(config)
    tokenizer, model = _load_tokenizer_and_model(config, device)
    features: list[torch.Tensor] = []
    batch_size = int(cfg["batch_size"])
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        attention_mask = encoded["attention_mask"].bool()
        positions = torch.arange(hidden.size(1), device=device).unsqueeze(0)
        token_counts = attention_mask.sum(dim=1, keepdim=True)
        residue_mask = attention_mask & (positions > 0) & (positions < token_counts - 1)
        pooled = (hidden * residue_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / residue_mask.sum(dim=1, keepdim=True).clamp_min(1)
        features.append(pooled.cpu())
    return torch.cat(features, dim=0)


def embed_sequences(
    sequences: Iterable[str],
    config: dict,
    cache_name: str | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    cleaned = [clean_sequence(sequence) for sequence in sequences]
    cleaned = [sequence for sequence in cleaned if sequence]
    if not cleaned:
        raise ValueError("No valid peptide sequences were provided for ESM-2 embedding.")

    cache_path = _cache_path(config, cleaned, cache_name)
    cfg = esm_config(config)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("sequences") == cleaned and cached.get("model_name") == cfg["model_name"]:
            return cached["embeddings"].float()

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = str(cfg["embedding_mode"]).lower()
    if mode == "contextual":
        embeddings = _contextual_features(cleaned, config, device)
    elif mode == "token":
        embeddings = _token_embedding_features(cleaned, config, device)
    else:
        raise ValueError("esm.embedding_mode must be either 'token' or 'contextual'.")

    torch.save(
        {
            "model_name": cfg["model_name"],
            "embedding_mode": mode,
            "sequences": cleaned,
            "embeddings": embeddings.float(),
        },
        cache_path,
    )
    return embeddings.float()
