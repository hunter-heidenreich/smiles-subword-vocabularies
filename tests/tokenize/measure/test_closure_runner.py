"""Tests for ``closure_runner`` (per-arm vocabulary-only build).

The runner reads only ``tokenizer.json`` via :func:`glyph_tuple_map`; here we
stub that with a fixture vocabulary so the closure counting is exercised
without a real artifact.
"""

from __future__ import annotations

import pytest

from smiles_subword.tokenize.measure.closure import runner as closure_runner


def test_run_arm_closure_uses_glyph_tuple_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = {
        0: ("C",),
        1: ("O",),
        2: ("C", "C"),
        3: ("C", "C", "C"),  # binary-closed via (C, CC)
        4: ("C", "O", "N"),  # orphan: no CO/ON/CON in vocab
    }
    monkeypatch.setattr(
        closure_runner, "glyph_tuple_map", lambda _artifact_dir, _arm: fixture
    )

    arm = closure_runner.run_arm_closure(
        cell_id="pubchem__smirk_unigram_v256_nmb",
        corpus="pubchem",
        name="smirk_unigram_v256_nmb",
        arm="unigram",
        boundary="nmb",
        vocab_size=256,
        training_corpus_sha="sha-x",
    )

    assert arm.arm == "unigram"
    assert arm.training_corpus_sha == "sha-x"
    assert arm.n_multi == 3  # CC, CCC, CON
    assert arm.n_ge3 == 2  # CCC, CON
    # CC closed (C+C), CCC closed (C+CC), CON not closed -> 2/3.
    assert arm.c_bin == pytest.approx(2 / 3)
    # CON is the only orphan among the two length->=3 pieces.
    assert arm.c_orph == pytest.approx(0.5)
