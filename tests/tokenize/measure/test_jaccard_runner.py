"""Tests for ``jaccard_runner`` (held-out J_w + training-dir resolution).

The glyph-tuple/glyph-count builders the runner uses live in ``_glyphmap`` and
are covered by ``test_glyphmap``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from smiles_subword.tokenize.measure.jaccard import runner as jaccard_runner
from smiles_subword.tokenize.measure.jaccard.inventory import (
    ChunkInventory,
    StructuralSplit,
)
from smiles_subword.tokenize.measure.jaccard.math import JwMoleculeData, bootstrap_seed
from smiles_subword.tokenize.measure.jaccard.runner import (
    build_jw_data,
    resolve_training_dir,
    run_arm_jaccard,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class _FakeAdapter:
    def __init__(self, enc: dict[str, list[int]]) -> None:
        self._enc = enc

    def encode_batch(
        self, batch: list[str], *, add_special_tokens: bool = False
    ) -> list[list[int]]:
        return [self._enc[s] for s in batch]


class TestBuildJwData:
    def test_counts_only_multi_glyph_emissions_per_molecule(self) -> None:
        glyph_tuple_by_id = {1: ("C",), 2: ("C", "C"), 3: ("c", "c")}
        adapter = _FakeAdapter(
            enc={"m0": [1, 2, 2], "deadzone": [3], "absorption": [1]}
        )

        jw = build_jw_data(
            adapter, ["m0", "deadzone", "absorption"], glyph_tuple_by_id, batch_size=2
        )

        assert jw.n_molecules == 3
        assert jw.total_emitted == 3
        counts = {
            jw.local_tuples[s]: int(c)
            for s, c in zip(jw.sub_local, jw.count, strict=True)
        }
        assert counts == {("C", "C"): 2, ("c", "c"): 1}

    def test_molecule_with_no_multi_glyph_still_counts(self) -> None:
        glyph_tuple_by_id = {1: ("C",), 2: ("C", "C")}
        adapter = _FakeAdapter(enc={"m0": [1, 1], "deadzone": [2]})

        jw = build_jw_data(adapter, ["m0", "deadzone"], glyph_tuple_by_id)

        assert jw.n_molecules == 2
        assert jw.total_emitted == 1


class TestRunArmJaccard:
    """The per-arm assembly: a multi-glyph-only vocab (atomic base excluded), the
    structural split from the *training* inventory, and J_w data from the
    *held-out* split. Driven through seams so the wiring is pinned locally.
    """

    def _run(
        self, monkeypatch: pytest.MonkeyPatch, *, glyph_map: dict, held_out: list[str]
    ) -> tuple[object, dict]:
        split = StructuralSplit(
            structural=frozenset({("C", "C")}),
            bracket_internal=frozenset({("c", "c")}),
            unseen=frozenset(),
        )
        inv = ChunkInventory(
            training_corpus_sha="t",
            bracket_chunks=("[X]", "[Y]"),
            nonbracket_chunks=("AB",),
            n_molecules_scanned=1,
            nonbracket_cap_bound=False,
        )
        monkeypatch.setattr(
            jaccard_runner, "tokenizer_artifact_dir", lambda c, n: Path("/art") / c / n
        )
        monkeypatch.setattr(jaccard_runner, "glyph_tuple_map", lambda _d, _a: glyph_map)
        monkeypatch.setattr(
            jaccard_runner, "resolve_training_dir", lambda c, n: Path("/train")
        )
        monkeypatch.setattr(
            jaccard_runner, "get_or_build_inventory", lambda *a, **k: inv
        )
        monkeypatch.setattr(jaccard_runner, "classify_subwords", lambda *a, **k: split)
        monkeypatch.setattr(jaccard_runner, "eval_split_sha", lambda _c: "eval-sha")
        monkeypatch.setattr(
            jaccard_runner,
            "iter_test_split",
            lambda _c, *, limit_molecules=None: iter(held_out),
        )

        captured: dict = {}

        def spy_build(
            _adapter: object,
            smiles: Iterable[str],
            glyph_tuple_by_id: dict,
            *,
            batch_size: int = 0,
        ) -> JwMoleculeData:
            captured["smiles"] = list(smiles)
            captured["glyph_map"] = glyph_tuple_by_id
            return JwMoleculeData(
                n_molecules=len(captured["smiles"]),
                mol_idx=np.asarray([], dtype=np.int64),
                sub_local=np.asarray([], dtype=np.int64),
                count=np.asarray([], dtype=np.float64),
                local_tuples=(),
            )

        monkeypatch.setattr(jaccard_runner, "build_jw_data", spy_build)

        result = run_arm_jaccard(
            object(),
            cell_id="corpus__smirk_gpe_v256_nmb",
            corpus="corpus",
            name="smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="t",
            inventory_cache_path=Path("/cache.json"),
        )
        return result, captured

    def test_multi_subwords_exclude_the_single_glyph_atomic_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # glyph map carries a single-glyph atom and two multi-glyph pieces; only
        # the len>=2 pieces are the comparable vocabulary.
        result, _ = self._run(
            monkeypatch,
            glyph_map={0: ("C",), 1: ("C", "C"), 2: ("c", "c")},
            held_out=["m0"],
        )

        assert result.multi_subwords == frozenset({("C", "C"), ("c", "c")})
        assert ("C",) not in result.multi_subwords

    def test_jw_is_built_from_the_held_out_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The J_w emission data must stream the held-out test split, not training:
        # a swap would silently turn J_w into a training-weighted Jaccard.
        held_out = ["heldout_a", "heldout_b", "heldout_c"]
        result, captured = self._run(
            monkeypatch,
            glyph_map={0: ("C",), 1: ("C", "C")},
            held_out=held_out,
        )

        assert captured["smiles"] == held_out
        assert result.jw.n_molecules == 3
        # build_jw_data needs the *full* id->tuple map (it filters emissions itself).
        assert captured["glyph_map"] == {0: ("C",), 1: ("C", "C")}

    def test_structural_split_and_provenance_pass_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, _ = self._run(
            monkeypatch,
            glyph_map={1: ("C", "C"), 2: ("c", "c")},
            held_out=["m0"],
        )

        assert result.structural_subwords == frozenset({("C", "C")})
        assert result.bracket_internal_subwords == frozenset({("c", "c")})
        assert result.unseen_subwords == frozenset()
        assert result.n_distinct_bracket_chunks == 2
        assert result.n_distinct_nonbracket_chunks == 1
        assert result.eval_split_sha == "eval-sha"
        assert result.bootstrap_seed == bootstrap_seed("corpus__smirk_gpe_v256_nmb")


class TestResolveTrainingDir:
    def test_grid_cell_uses_headline_training_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(jaccard_runner, "load_extras_manifest", list)

        out = resolve_training_dir("pubchem", "smirk_gpe_v256_nmb")

        assert out.parts[-2:] == ("canon_dedup_v1", "train")
        assert "pubchem" in out.parts

    def test_extras_subsample_cell_uses_subsample_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from smiles_subword.tokenize.extras import load_extras_manifest

        cells = [c for c in load_extras_manifest() if c.training_subdir is not None]
        if not cells:
            pytest.skip("no extras cell with a training_subdir in the manifest")
        cell = cells[0]

        out = resolve_training_dir(cell.corpus, cell.name)

        assert "canon_dedup_v1_extras" in out.parts
        assert cell.training_subdir in out.parts
