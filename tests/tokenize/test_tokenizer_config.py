"""Tests for the Stage 5 tokenizer config: naming rules + cross-field validation.

The naming trio (`algo_to_engine_tag` / `algo_to_kind` / `cell_artifact_name`)
is the single source of the on-disk cell-naming convention; pin it in isolation
rather than only transitively through GridCell/ExtrasCell. `TokenizerConfig`'s
`_validate_kind_requirements` is a multi-branch cross-field validator whose
raise paths were otherwise unexercised — pin each one.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from smiles_subword.config import (
    TokenizerConfig,
    algo_to_engine_tag,
    algo_to_kind,
    cell_artifact_name,
)


def _payload(**overrides: object) -> dict[str, object]:
    """A minimal valid `smirk_base` payload; override per case."""
    payload: dict[str, object] = {
        "name": "cell",
        "kind": "smirk_base",
        "output_dir": "artifacts/cell",
    }
    payload.update(overrides)
    return payload


def _trainable(kind: str, **overrides: object) -> dict[str, object]:
    """A minimal valid trainable (`smirk_gpe`/`smirk_unigram`) payload."""
    return _payload(
        kind=kind,
        vocab_size=256,
        training_input="data/processed/pubchem/canon_dedup_v1/train",
        **overrides,
    )


# --- naming trio -------------------------------------------------------------


def test_algo_to_engine_tag() -> None:
    assert algo_to_engine_tag("bpe") == "gpe"
    assert algo_to_engine_tag("unigram") == "unigram"


def test_algo_to_kind() -> None:
    assert algo_to_kind("bpe") == "smirk_gpe"
    assert algo_to_kind("unigram") == "smirk_unigram"


def test_cell_artifact_name_without_suffix() -> None:
    assert cell_artifact_name("bpe", 256, "nmb") == "smirk_gpe_v256_nmb"
    assert cell_artifact_name("unigram", 1024, "mb") == "smirk_unigram_v1024_mb"


def test_cell_artifact_name_with_suffix() -> None:
    assert (
        cell_artifact_name("bpe", 256, "nmb", suffix="seedA")
        == "smirk_gpe_v256_nmb_seedA"
    )


# --- TokenizerConfig validator: happy paths ----------------------------------


def test_base_kind_needs_no_vocab_or_training_input() -> None:
    cfg = TokenizerConfig.model_validate(_payload())
    assert cfg.kind == "smirk_base"
    assert cfg.vocab_size is None
    assert cfg.training_input is None


@pytest.mark.parametrize("kind", ["smirk_gpe", "smirk_unigram"])
def test_trainable_kind_accepts_vocab_and_training_input(kind: str) -> None:
    cfg = TokenizerConfig.model_validate(_trainable(kind))
    assert cfg.kind == kind
    assert cfg.vocab_size == 256


# --- TokenizerConfig validator: raise paths ----------------------------------


@pytest.mark.parametrize("kind", ["smirk_gpe", "smirk_unigram"])
def test_trainable_kind_requires_vocab_size(kind: str) -> None:
    payload = _trainable(kind)
    del payload["vocab_size"]
    with pytest.raises(ValidationError, match="requires vocab_size"):
        TokenizerConfig.model_validate(payload)


@pytest.mark.parametrize("kind", ["smirk_gpe", "smirk_unigram"])
def test_trainable_kind_requires_training_input(kind: str) -> None:
    payload = _trainable(kind)
    del payload["training_input"]
    with pytest.raises(ValidationError, match="requires training_input"):
        TokenizerConfig.model_validate(payload)


def test_ref_artifact_dir_rejected_for_non_gpe() -> None:
    payload = _trainable("smirk_unigram", ref_artifact_dir="artifacts/ref")
    with pytest.raises(ValidationError, match="ref_artifact_dir is only meaningful"):
        TokenizerConfig.model_validate(payload)


@pytest.mark.parametrize("knob", [{"merge_brackets": True}, {"split_structure": False}])
def test_boundary_knobs_rejected_for_base_kind(knob: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="merge_brackets / split_structure"):
        TokenizerConfig.model_validate(_payload(**knob))


@pytest.mark.parametrize(
    "knob",
    [
        {"seed_size": 10},
        {"max_piece_length": 5},
        {"n_sub_iterations": 2},
        {"shrinking_factor": 0.5},
    ],
)
def test_unigram_knobs_rejected_for_non_unigram(knob: dict[str, object]) -> None:
    payload = _trainable("smirk_gpe", **knob)
    match = "only meaningful for kind='smirk_unigram'"
    with pytest.raises(ValidationError, match=match):
        TokenizerConfig.model_validate(payload)


def test_scaffold_log_rejected_for_non_gpe() -> None:
    payload = _trainable("smirk_unigram", scaffold_log=True)
    with pytest.raises(ValidationError, match="scaffold_log is only meaningful"):
        TokenizerConfig.model_validate(payload)
