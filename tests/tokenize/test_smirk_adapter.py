"""Tests for the Smirk adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from smirk import SmirkTokenizerFast

from smiles_subword.tokenize import (
    ATOMIC_VOCAB_SIZE,
    SmirkAdapter,
    Tokenizer,
    materialize_smiles_txt,
    training_corpus_sha,
)
from smiles_subword.tokenize.adapters.smirk_unigram import (
    DEFAULT_MAX_PIECE_LENGTH,
    DEFAULT_N_SUB_ITERATIONS,
    DEFAULT_SEED_SIZE,
    DEFAULT_SHRINKING_FACTOR,
    UnigramSmirkAdapter,
)

ASPIRIN: str = "CC(=O)Oc1ccccc1C(=O)O"

_SAMPLE_SMILES: tuple[str, ...] = (
    "CCO",
    "CC(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1ccccc1",
    "Cc1ccc(C)cc1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "CN1CCC[C@H]1c1cccnc1",
    "OC(=O)c1ccccc1O",
    "Nc1ccc(S(N)(=O)=O)cc1",
    "ClC(Cl)(Cl)Cl",
    "BrCCBr",
    "FC(F)(F)C(=O)O",
    "C1CCCCC1",
    "C1CCNCC1",
    "C1CCOC1",
    "OCC(O)C(O)C(O)C(O)CO",
    "CC(=O)OCC(COC(C)=O)OC(C)=O",
)


@pytest.fixture(scope="module")
def synth_corpus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small SMILES corpus on disk; large enough for GPE to find merges."""
    tmp = tmp_path_factory.mktemp("smirk_corpus")
    path = tmp / "corpus.smi"
    path.write_text("\n".join(list(_SAMPLE_SMILES) * 100))
    return path


@pytest.fixture(scope="module")
def gpe_v200(synth_corpus_path: Path) -> SmirkAdapter:
    """A GPE-trained Smirk adapter at V≈200 — module-scoped to share train cost."""
    return SmirkAdapter.train_gpe(
        [synth_corpus_path],
        name="smirk_gpe_v200",
        vocab_size=200,
        min_frequency=2,
    )


@pytest.fixture(scope="module")
def unigram_v200(synth_corpus_path: Path) -> UnigramSmirkAdapter:
    """A Unigram-trained adapter at the same V≈200 target — module-scoped."""
    return UnigramSmirkAdapter.train_unigram(
        [synth_corpus_path],
        name="smirk_unigram_v200",
        vocab_size=200,
    )


# A deliberately glyph-narrow corpus: only C, O, =, (), and ring-1 glyphs.
# No N/S/P/F/Cl/Br/I/B, no aromatic c/n/o/s, no brackets/charges/stereo — so
# most of the OpenSMILES base is *unsupported* by the training data. Used to
# certify that both trainers still retain the full base (coverage guarantee).
_NARROW_SMILES: tuple[str, ...] = (
    "CCO",
    "CCCC",
    "CC(=O)O",
    "C1CCCCC1",
    "OCCO",
    "CCCCCCCC",
    "CC(C)C",
    "O=CCC=O",
)

# Molecules using base glyphs absent from _NARROW_SMILES; a tokenizer that
# dropped any base glyph would emit [UNK] / fail to round-trip these.
_OUT_OF_CORPUS_PROBES: tuple[str, ...] = (
    "BrCCBr",
    "FC(F)(F)F",
    "ClCCl",
    "CCN",
    "CCS",
    "c1ccccc1",
    "O=P(O)(O)O",
    "[NH4+]",
)


@pytest.fixture(scope="module")
def narrow_corpus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A glyph-narrow SMILES corpus on disk (C/O aliphatic only)."""
    tmp = tmp_path_factory.mktemp("smirk_narrow_corpus")
    path = tmp / "corpus.smi"
    path.write_text("\n".join(list(_NARROW_SMILES) * 80))
    return path


@pytest.fixture(scope="module")
def unigram_narrow(narrow_corpus_path: Path) -> UnigramSmirkAdapter:
    """Unigram trained on a glyph-narrow corpus (C/O aliphatic only)."""
    return UnigramSmirkAdapter.train_unigram(
        [narrow_corpus_path], name="smirk_unigram_narrow", vocab_size=256
    )


@pytest.fixture(scope="module")
def gpe_narrow(narrow_corpus_path: Path) -> SmirkAdapter:
    """GPE trained on a glyph-narrow corpus (C/O aliphatic only)."""
    return SmirkAdapter.train_gpe(
        [narrow_corpus_path], name="smirk_gpe_narrow", vocab_size=256, min_frequency=2
    )


# Bracket-rich corpus: charged / stereo / isotope / aromatic-NH bracket atoms
# kept adjacent to mergeable neighbors, so cross-bracket bigrams are frequent
# enough that the MB trainer will merge across the bracket boundary.
_BRACKET_RICH_SMILES: tuple[str, ...] = (
    "C[NH3+]",
    "CC[O-]",
    "C[C@@H](O)C",
    "c1cc[nH]c1",
    "C[N+](C)(C)C",
    "O=C[O-]",
    "C[NH3+]CC[O-]",
    "C[C@H](N)C(=O)[O-]",
    "[NH4+]",
    "C[O-]",
)


@pytest.fixture(scope="module")
def bracket_corpus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A bracket-atom-rich SMILES corpus on disk."""
    tmp = tmp_path_factory.mktemp("smirk_bracket_corpus")
    path = tmp / "corpus.smi"
    path.write_text("\n".join(list(_BRACKET_RICH_SMILES) * 100))
    return path


@pytest.fixture(scope="module")
def gpe_nmb(bracket_corpus_path: Path) -> SmirkAdapter:
    """GPE under no-merge-brackets (opaque bracketed atoms)."""
    return SmirkAdapter.train_gpe(
        [bracket_corpus_path],
        name="smirk_gpe_nmb",
        vocab_size=240,
        min_frequency=2,
        merge_brackets=False,
    )


@pytest.fixture(scope="module")
def gpe_mb(bracket_corpus_path: Path) -> SmirkAdapter:
    """GPE under merge-brackets (permeable bracketed atoms)."""
    return SmirkAdapter.train_gpe(
        [bracket_corpus_path],
        name="smirk_gpe_mb",
        vocab_size=240,
        min_frequency=2,
        merge_brackets=True,
    )


@pytest.fixture
def base_tok() -> SmirkAdapter:
    return SmirkAdapter.atomic()


class TestAtomicBaseline:
    """``SmirkAdapter.atomic()`` returns the stock 159-token tokenizer."""

    def test_satisfies_runtime_tokenizer_protocol(self, base_tok: SmirkAdapter) -> None:
        assert isinstance(base_tok, Tokenizer)

    def test_vocab_size_matches_atomic_constant(self, base_tok: SmirkAdapter) -> None:
        assert base_tok.vocab_size == ATOMIC_VOCAB_SIZE

    def test_base_kind_is_smirk_base(self, base_tok: SmirkAdapter) -> None:
        assert base_tok.base_kind == "smirk_base"

    def test_n_merges_is_none(self, base_tok: SmirkAdapter) -> None:
        assert base_tok.n_merges is None

    def test_special_token_ids_are_distinct(self, base_tok: SmirkAdapter) -> None:
        assert len({base_tok.bos_id, base_tok.eos_id, base_tok.pad_id}) == 3

    def test_unk_id_is_an_int(self, base_tok: SmirkAdapter) -> None:
        assert isinstance(base_tok.unk_id, int)

    def test_default_name_is_smirk_base(self, base_tok: SmirkAdapter) -> None:
        assert base_tok.name == "smirk_base"


class TestRoundTrip:
    """Encode/decode preserves canonical SMILES after whitespace strip."""

    def test_aspirin_decodes_back_to_input_on_atomic(
        self, base_tok: SmirkAdapter
    ) -> None:
        ids = base_tok.encode(ASPIRIN)
        assert base_tok.decode(ids) == ASPIRIN

    def test_aspirin_decodes_back_to_input_on_gpe(self, gpe_v200: SmirkAdapter) -> None:
        ids = gpe_v200.encode(ASPIRIN)
        assert gpe_v200.decode(ids) == ASPIRIN

    def test_encode_with_special_tokens_brackets_sequence(
        self, base_tok: SmirkAdapter
    ) -> None:
        ids = base_tok.encode(ASPIRIN, add_special_tokens=True)
        assert ids[0] == base_tok.bos_id
        assert ids[-1] == base_tok.eos_id

    def test_token_to_id_returns_none_for_unknown(self, base_tok: SmirkAdapter) -> None:
        assert base_tok.token_to_id("definitely-not-a-token") is None

    def test_id_to_token_round_trips_for_atomic_C(self, base_tok: SmirkAdapter) -> None:
        idx = base_tok.token_to_id("C")
        assert idx is not None
        assert base_tok.id_to_token(idx) == "C"


class TestSaveLoad:
    """The artifact contract round-trips."""

    def test_save_emits_meta_yaml_for_atomic(
        self, base_tok: SmirkAdapter, tmp_path: Path
    ) -> None:
        base_tok.save(tmp_path)

        assert (tmp_path / "tokenizer.json").exists()
        assert (tmp_path / "tokenizer_config.json").exists()
        assert (tmp_path / "special_tokens_map.json").exists()
        assert (tmp_path / "meta.yaml").exists()
        assert not (tmp_path / "merges.txt").exists()

    def test_save_emits_merges_txt_for_gpe(
        self, gpe_v200: SmirkAdapter, tmp_path: Path
    ) -> None:
        gpe_v200.save(tmp_path)

        merges_text = (tmp_path / "merges.txt").read_text().splitlines()
        assert merges_text[0] == "#version: 0.2"
        assert all(len(line.split(" ")) == 2 for line in merges_text[1:])
        assert len(merges_text) - 1 == gpe_v200.n_merges

    def test_meta_yaml_carries_required_fields(
        self, gpe_v200: SmirkAdapter, tmp_path: Path
    ) -> None:
        gpe_v200.save(tmp_path)
        meta = yaml.safe_load((tmp_path / "meta.yaml").read_text())

        assert meta["name"] == "smirk_gpe_v200"
        assert meta["base_kind"] == "smirk_gpe"
        assert meta["vocab_size"] == gpe_v200.vocab_size
        assert meta["n_merges"] == gpe_v200.n_merges

    def test_round_trip_preserves_vocab_size_and_encoding(
        self, gpe_v200: SmirkAdapter, tmp_path: Path
    ) -> None:
        gpe_v200.save(tmp_path)
        restored = SmirkAdapter.load(tmp_path)

        assert restored.vocab_size == gpe_v200.vocab_size
        assert restored.encode(ASPIRIN) == gpe_v200.encode(ASPIRIN)

    def test_round_trip_preserves_meta_attributes(
        self, gpe_v200: SmirkAdapter, tmp_path: Path
    ) -> None:
        gpe_v200.save(tmp_path)
        restored = SmirkAdapter.load(tmp_path)

        assert restored.name == gpe_v200.name
        assert restored.base_kind == gpe_v200.base_kind
        assert restored.n_merges == gpe_v200.n_merges


class TestRefChainedTrajectory:
    """``ref=`` chaining produces a monotonic merge trajectory."""

    def test_chained_v_grows_monotonically(self, synth_corpus_path: Path) -> None:
        v1 = SmirkAdapter.train_gpe(
            [synth_corpus_path], name="v1", vocab_size=180, min_frequency=2
        )
        v2 = SmirkAdapter.train_gpe(
            [synth_corpus_path],
            name="v2",
            vocab_size=200,
            min_frequency=2,
            ref=v1,
        )

        assert v1.vocab_size > ATOMIC_VOCAB_SIZE
        assert v2.vocab_size > v1.vocab_size
        assert v1.n_merges is not None
        assert v2.n_merges is not None
        assert v2.n_merges > v1.n_merges

    def test_chained_v2_extends_v1_merge_list(
        self, synth_corpus_path: Path, tmp_path: Path
    ) -> None:
        v1 = SmirkAdapter.train_gpe(
            [synth_corpus_path], name="v1", vocab_size=180, min_frequency=2
        )
        v2 = SmirkAdapter.train_gpe(
            [synth_corpus_path],
            name="v2",
            vocab_size=200,
            min_frequency=2,
            ref=v1,
        )
        v1_path = tmp_path / "v1"
        v2_path = tmp_path / "v2"
        v1.save(v1_path)
        v2.save(v2_path)
        v1_merges = (v1_path / "merges.txt").read_text().splitlines()[1:]
        v2_merges = (v2_path / "merges.txt").read_text().splitlines()[1:]

        assert v2_merges[: len(v1_merges)] == v1_merges


class TestTrainGpeRejectsBelowAtomic:
    """Defensive check: GPE target below the atomic floor is a config bug."""

    def test_rejects_vocab_size_below_atomic(self, synth_corpus_path: Path) -> None:
        with pytest.raises(ValueError, match="atomic baseline"):
            SmirkAdapter.train_gpe(
                [synth_corpus_path],
                name="too_small",
                vocab_size=ATOMIC_VOCAB_SIZE - 1,
            )


class TestMaterializeSmilesTxt:
    """``materialize_smiles_txt`` flattens a Parquet shard set to a .smi text dump."""

    def test_writes_one_smiles_per_line_in_shard_order(self, tmp_path: Path) -> None:
        parquet_dir = tmp_path / "stage"
        parquet_dir.mkdir()
        for i, slice_ in enumerate([_SAMPLE_SMILES[:5], _SAMPLE_SMILES[5:10]]):
            table = pa.table({"smiles": list(slice_)})
            pq.write_table(table, parquet_dir / f"sub_v1-{i:05d}.parquet")
        out = tmp_path / "out.smi"
        materialize_smiles_txt(parquet_dir, out)

        assert out.read_text().splitlines() == list(_SAMPLE_SMILES[:10])


class TestTrainingCorpusSha:
    """``training_corpus_sha`` is a deterministic 32-hex fingerprint."""

    @staticmethod
    def _write_manifest(parquet_dir: Path, shard_shas: list[str]) -> None:
        payload = {
            "schema": "sub_v1",
            "shards": [
                {
                    "file": f"sub_v1-{i:05d}.parquet",
                    "sha256": sha,
                    "n_rows": 1,
                    "n_bytes": 1,
                }
                for i, sha in enumerate(shard_shas)
            ],
        }
        with (parquet_dir / "MANIFEST.yaml").open("w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)

    def test_returns_32_hex_chars(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path, ["a" * 64, "b" * 64, "c" * 64])

        sha = training_corpus_sha(tmp_path)

        assert len(sha) == 32
        assert all(c in "0123456789abcdef" for c in sha)

    def test_is_deterministic(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path, ["a" * 64, "b" * 64, "c" * 64])

        assert training_corpus_sha(tmp_path) == training_corpus_sha(tmp_path)

    def test_invariant_under_manifest_shard_order(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        self._write_manifest(a, ["1" * 64, "2" * 64, "3" * 64])
        self._write_manifest(b, ["3" * 64, "1" * 64, "2" * 64])

        assert training_corpus_sha(a) == training_corpus_sha(b)

    def test_changes_when_a_shard_sha_changes(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        self._write_manifest(a, ["1" * 64, "2" * 64])
        self._write_manifest(b, ["1" * 64, "9" * 64])

        assert training_corpus_sha(a) != training_corpus_sha(b)

    def test_raises_on_empty_shards(self, tmp_path: Path) -> None:
        self._write_manifest(tmp_path, [])

        with pytest.raises(ValueError, match="no shards"):
            training_corpus_sha(tmp_path)


_LAYER_B_EXPECTED: dict[str, list[tuple[str, tuple[int, int]]]] = {
    "CC": [("CC", (0, 2))],
    "C.C": [("C", (0, 1)), (".", (1, 2)), ("C", (2, 3))],
    "C(C)": [("C", (0, 1)), ("(", (1, 2)), ("C", (2, 3)), (")", (3, 4))],
    "C1ccccc1C": [
        ("C", (0, 1)),
        ("1", (1, 2)),
        ("ccccc", (2, 7)),
        ("1", (7, 8)),
        ("C", (8, 9)),
    ],
    "C[13C]": [("C", (0, 1)), ("[13C]", (1, 6))],
    "CC(=O)O": [
        ("CC", (0, 2)),
        ("(", (2, 3)),
        ("=O", (3, 5)),
        (")", (5, 6)),
        ("O", (6, 7)),
    ],
}


class TestPretokenizeLayerB:
    """``pretokenize_layer_b`` exposes the Layer-B chunker for absorption."""

    @pytest.mark.parametrize(
        ("smi", "expected"),
        [pytest.param(s, e, id=s) for s, e in _LAYER_B_EXPECTED.items()],
    )
    def test_atomic_chunks_match_pinned_boundaries(
        self,
        base_tok: SmirkAdapter,
        smi: str,
        expected: list[tuple[str, tuple[int, int]]],
    ) -> None:
        chunks = base_tok.pretokenize_layer_b(smi)

        assert [(c, tuple(o)) for c, o in chunks] == expected

    @pytest.mark.parametrize(
        "smi",
        [
            "CCOc1ccc(C(=O)c2ccccc2)cc1",  # rings + branches
            "C[C@@H](N)C(=O)O",  # stereo bracket atom
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin: branches, ring, carbonyls
            "[13C]C[NH3+]",  # isotope + charged bracket atoms
            "C1CC2CCC1CC2",  # fused-ring branch points
        ],
    )
    def test_both_arms_pretokenize_identically_to_the_baseline(
        self,
        base_tok: SmirkAdapter,
        gpe_v200: SmirkAdapter,
        unigram_v200: UnigramSmirkAdapter,
        smi: str,
    ) -> None:
        # The controlled-comparison premise: at a matched coordinate the
        # *only* cross-arm difference is the selection algorithm — both arms
        # consume an identical Layer-B glyph stream. Pre-tokenization is upstream
        # of merge/prune, so the baseline and both trained arms must agree
        # exactly, including across bracket / ring / branch boundaries.
        base_chunks = base_tok.pretokenize_layer_b(smi)
        assert gpe_v200.pretokenize_layer_b(smi) == base_chunks
        assert unigram_v200.pretokenize_layer_b(smi) == base_chunks

    @pytest.mark.parametrize("smi", list(_LAYER_B_EXPECTED))
    def test_spans_cover_input_contiguously(
        self, base_tok: SmirkAdapter, smi: str
    ) -> None:
        from itertools import pairwise

        chunks = base_tok.pretokenize_layer_b(smi)
        spans = [tuple(o) for _, o in chunks]

        assert "".join(c for c, _ in chunks) == smi
        assert spans[0][0] == 0
        assert spans[-1][1] == len(smi)
        for (_, end), (start, _) in pairwise(spans):
            assert end == start

    def test_empty_input_yields_no_chunks(self, base_tok: SmirkAdapter) -> None:
        assert base_tok.pretokenize_layer_b("") == []


class TestScaffoldLog:
    """``scaffold_log_path`` wiring (scaffold instrumentation)."""

    def test_absent_when_path_unset(
        self, synth_corpus_path: Path, tmp_path: Path
    ) -> None:
        tok = SmirkAdapter.train_gpe(
            [synth_corpus_path], name="no_log", vocab_size=200, min_frequency=2
        )
        tok.save(tmp_path)

        assert not (tmp_path / "scaffold.jsonl").exists()
        meta = yaml.safe_load((tmp_path / "meta.yaml").read_text())
        assert meta.get("scaffold_log_sha") is None

    def test_written_when_path_set(
        self, synth_corpus_path: Path, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "scaffold.jsonl"
        tok = SmirkAdapter.train_gpe(
            [synth_corpus_path],
            name="with_log",
            vocab_size=200,
            min_frequency=2,
            scaffold_log_path=log_path,
        )
        tok.save(tmp_path)

        assert log_path.is_file()
        lines = log_path.read_text().splitlines()
        assert lines[0].startswith('{"format":"smirk-scaffold-log/v1"')
        assert any('"step":' in line for line in lines[1:])

    def test_sha_recorded_in_meta(
        self, synth_corpus_path: Path, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "scaffold.jsonl"
        tok = SmirkAdapter.train_gpe(
            [synth_corpus_path],
            name="sha_test",
            vocab_size=200,
            min_frequency=2,
            scaffold_log_path=log_path,
        )
        tok.save(tmp_path)

        meta = yaml.safe_load((tmp_path / "meta.yaml").read_text())
        expected = hashlib.sha256(log_path.read_bytes()).hexdigest()
        assert meta["scaffold_log_sha"] == expected

    def test_byte_identical_tokenizer_with_and_without_log(
        self, synth_corpus_path: Path, tmp_path: Path
    ) -> None:
        """Scaffold contract: logging-only does not alter merge selection."""
        no_log_dir = tmp_path / "no_log"
        log_dir = tmp_path / "with_log"
        no_log_dir.mkdir()
        log_dir.mkdir()

        no_log = SmirkAdapter.train_gpe(
            [synth_corpus_path], name="bi_no_log", vocab_size=200, min_frequency=2
        )
        no_log.save(no_log_dir)
        with_log = SmirkAdapter.train_gpe(
            [synth_corpus_path],
            name="bi_no_log",
            vocab_size=200,
            min_frequency=2,
            scaffold_log_path=log_dir / "scaffold.jsonl",
        )
        with_log.save(log_dir)

        assert (no_log_dir / "tokenizer.json").read_bytes() == (
            log_dir / "tokenizer.json"
        ).read_bytes()
        assert (no_log_dir / "merges.txt").read_bytes() == (
            log_dir / "merges.txt"
        ).read_bytes()

    def test_load_round_trips_scaffold_log_path(
        self, synth_corpus_path: Path, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "scaffold.jsonl"
        tok = SmirkAdapter.train_gpe(
            [synth_corpus_path],
            name="rt",
            vocab_size=200,
            min_frequency=2,
            scaffold_log_path=log_path,
        )
        tok.save(tmp_path)

        restored = SmirkAdapter.load(tmp_path)

        assert restored._scaffold_log_path == tmp_path / "scaffold.jsonl"


# Tail specials appended above the WordLevel base by both smirk trainers.
# ``[UNK]`` is excluded — it overlaps the atomic base at id 0 rather than
# sitting above it.
_TAIL_SPECIALS = ("[BOS]", "[EOS]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")

# All seven specials (the six tail specials plus [UNK]); excluded when isolating
# the OpenSMILES glyph base from a tokenizer's forward vocabulary.
_ALL_SPECIALS = frozenset((*_TAIL_SPECIALS, "[UNK]"))


def _base_glyph_tokens(adapter: SmirkAdapter | UnigramSmirkAdapter) -> set[str]:
    """The single-glyph OpenSMILES base of a tokenizer (forward vocab, no specials)."""
    return {t for t in adapter._tok.get_vocab() if t not in _ALL_SPECIALS}


def _merge_lines(adapter: SmirkAdapter, artifact_dir: Path) -> list[str]:
    """Save the GPE artifact and return its ``merges.txt`` rules (no header)."""
    adapter.save(artifact_dir)
    return (artifact_dir / "merges.txt").read_text().splitlines()[1:]


def _has_bracket(line: str) -> bool:
    return "[" in line or "]" in line


def _tail_specials_above_base(artifact_dir: Path) -> list[str]:
    """Contents of ``added_tokens`` whose id sits above ``model.vocab``."""
    payload = json.loads((artifact_dir / "tokenizer.json").read_text())
    n_base = len(payload["model"]["vocab"])
    return [a["content"] for a in payload["added_tokens"] if a["id"] >= n_base]


class TestVocabSizeSpecialsContract:
    """The shipped ``vocab_size`` / ``len(tok)`` contract for both arms.

    No artifact is post-train trimmed: each arm ships exactly what its trainer
    produces. Both append six tail specials above the WordLevel base, so the
    embedding axis ``len(tok)`` sits above the trained ``vocab_size``. The arms
    differ in what ``vocab_size`` *means*:

    - ``GpeTrainer.vocab_size`` targets the base exactly, so the artifact ships
      at ``len(tok) == vocab_size + n_tail_specials`` (a fixed +6 above the
      base) — this is the natural realized size, not an off-target overshoot.
    - HF ``UnigramTrainer.vocab_size`` is an *upper bound* the pruning loop
      generally undershoots.

    These pin the smirk-fork contract: a future smirk / HF ``tokenizers`` bump
    that changed how ``vocab_size`` counts specials would trip these tests.
    """

    def test_gpe_appends_six_tail_specials_above_base(
        self, gpe_v200: SmirkAdapter, tmp_path: Path
    ) -> None:
        gpe_v200.save(tmp_path)
        assert _tail_specials_above_base(tmp_path) == list(_TAIL_SPECIALS)

    def test_gpe_ships_len_six_above_vocab_size(self, gpe_v200: SmirkAdapter) -> None:
        # The natural realized size: the embedding axis is six tail specials
        # larger than the trained WordLevel base. No trim pins it to V.
        assert len(gpe_v200) == gpe_v200.vocab_size + len(_TAIL_SPECIALS)
        assert len(gpe_v200) > gpe_v200.vocab_size

    def test_unigram_does_not_overshoot_the_request(
        self, unigram_v200: UnigramSmirkAdapter
    ) -> None:
        # vocab_size is an upper bound the prune loop honours (here it
        # undershoots, capped by the corpus) — never above the request.
        assert unigram_v200.vocab_size <= 200

    def test_unigram_also_appends_specials_but_writes_no_merges(
        self, unigram_v200: UnigramSmirkAdapter, tmp_path: Path
    ) -> None:
        # Unigram appends the same six tail specials (so the asymmetry is
        # *not* "Unigram omits specials") but has no merge list.
        # Note: unlike GPE, Unigram also places ``[UNK]`` above its model
        # vocab rather than overlapping the base, so it carries seven tail
        # entries — hence a subset check on the six shared specials.
        unigram_v200.save(tmp_path)
        above = _tail_specials_above_base(tmp_path)
        assert set(_TAIL_SPECIALS).issubset(above)
        assert not (tmp_path / "merges.txt").exists()


class TestOpenSmilesBaseRetention:
    """Both arms retain 100% of the OpenSMILES base — the coverage guarantee.

    Smirk installs the full glyph base as length-1 pieces that neither trainer
    may prune, so coverage holds even when the training corpus exercises only a
    tiny subset of glyphs. These cells train on a deliberately glyph-narrow
    corpus (C/O aliphatic only) yet must still carry every base glyph and
    losslessly encode molecules built from glyphs absent from training. A
    tokenizer that dropped a base glyph would UNK those molecules and break the
    OpenSMILES-coverage claim rests on.
    """

    def test_gpe_retains_every_base_glyph(self, gpe_narrow: SmirkAdapter) -> None:
        atomic_base = _base_glyph_tokens(SmirkAdapter.atomic())
        assert atomic_base, "sanity: atomic baseline must expose its glyph base"
        missing = atomic_base - _base_glyph_tokens(gpe_narrow)
        assert missing == set(), f"GPE pruned base glyphs: {sorted(missing)}"

    def test_unigram_retains_every_base_glyph(
        self, unigram_narrow: UnigramSmirkAdapter
    ) -> None:
        atomic_base = _base_glyph_tokens(SmirkAdapter.atomic())
        missing = atomic_base - _base_glyph_tokens(unigram_narrow)
        assert missing == set(), f"Unigram pruned base glyphs: {sorted(missing)}"

    @pytest.mark.parametrize("smi", _OUT_OF_CORPUS_PROBES)
    def test_gpe_encodes_out_of_corpus_glyphs_without_unk(
        self, gpe_narrow: SmirkAdapter, smi: str
    ) -> None:
        ids = gpe_narrow.encode(smi)
        assert gpe_narrow.unk_id not in ids
        assert gpe_narrow.decode(ids) == smi

    @pytest.mark.parametrize("smi", _OUT_OF_CORPUS_PROBES)
    def test_unigram_encodes_out_of_corpus_glyphs_without_unk(
        self, unigram_narrow: UnigramSmirkAdapter, smi: str
    ) -> None:
        ids = unigram_narrow.encode(smi)
        assert unigram_narrow.unk_id not in ids
        assert unigram_narrow.decode(ids) == smi


class TestUnigramReferenceDefaults:
    """The Unigram knobs stay at the frozen reference values.

    "Both run at their reference implementations' training defaults, so neither
    is hand-tuned to favor the contrast". These defaults feed every
    Unigram artifact, and ``max_piece_length`` has already been changed once
    (128 → 16) — so a drift here would silently alter the whole arm and force a
    re-train. Pin them.
    """

    def test_defaults_match_frozen_reference_values(self) -> None:
        assert DEFAULT_SEED_SIZE == 1_000_000
        assert DEFAULT_MAX_PIECE_LENGTH == 16
        assert DEFAULT_N_SUB_ITERATIONS == 2
        assert DEFAULT_SHRINKING_FACTOR == 0.75

    def test_train_unigram_signature_uses_the_frozen_defaults(self) -> None:
        # The trainer entry point must default to the pinned constants, not
        # ad-hoc literals that could drift away from them.
        import inspect

        params = inspect.signature(UnigramSmirkAdapter.train_unigram).parameters
        assert params["seed_size"].default == DEFAULT_SEED_SIZE
        assert params["max_piece_length"].default == DEFAULT_MAX_PIECE_LENGTH
        assert params["n_sub_iterations"].default == DEFAULT_N_SUB_ITERATIONS
        assert params["shrinking_factor"].default == DEFAULT_SHRINKING_FACTOR

    def test_base_kind_is_smirk_unigram(
        self, unigram_v200: UnigramSmirkAdapter
    ) -> None:
        # Direct literal pin of the __init__ assignment (GPE has the analogous
        # test_base_kind_is_smirk_base); the unigram literal was only transitive.
        assert unigram_v200.base_kind == "smirk_unigram"

    def test_satisfies_runtime_tokenizer_protocol(
        self, unigram_v200: UnigramSmirkAdapter
    ) -> None:
        assert isinstance(unigram_v200, Tokenizer)


class TestBoundaryPolicySemantics:
    """The NMB / MB boundary lever does what the boundary policy specifies.

    Under **no-merge-brackets** a bracketed atom is opaque, so no derived merge
    may contain a ``[`` / ``]`` — the bracket boundary is impermeable. Under
    **merge-brackets** the boundary is permeable, so merges may cross into and
    out of bracketed atoms. ``merges.txt`` is the authoritative merge list; a
    regression that ignored the ``merge_brackets`` flag would either leak
    bracket merges into NMB or stop producing them under MB.
    """

    def test_nmb_emits_no_bracket_crossing_merge(
        self, gpe_nmb: SmirkAdapter, tmp_path: Path
    ) -> None:
        lines = _merge_lines(gpe_nmb, tmp_path)
        assert lines, "sanity: the bracket-rich corpus should yield some merges"
        offending = [ln for ln in lines if _has_bracket(ln)]
        assert offending == [], f"NMB leaked bracket-crossing merges: {offending}"

    def test_mb_permits_bracket_crossing_merges(
        self, gpe_mb: SmirkAdapter, tmp_path: Path
    ) -> None:
        lines = _merge_lines(gpe_mb, tmp_path)
        assert any(_has_bracket(ln) for ln in lines), (
            "MB should merge across bracket boundaries on a bracket-rich corpus"
        )

    def test_boundary_flag_changes_the_learned_vocabulary(
        self, gpe_nmb: SmirkAdapter, gpe_mb: SmirkAdapter, tmp_path: Path
    ) -> None:
        # The lever has a real effect: same corpus + V, the two policies learn
        # different merge sets.
        nmb = set(_merge_lines(gpe_nmb, tmp_path / "nmb"))
        mb = set(_merge_lines(gpe_mb, tmp_path / "mb"))
        assert nmb != mb

    def test_meta_yaml_carries_boundary_flags(
        self, gpe_nmb: SmirkAdapter, gpe_mb: SmirkAdapter, tmp_path: Path
    ) -> None:
        # The NMB/MB axis is provenance: meta.yaml must record which policy each
        # artifact was trained under (the Unigram knobs are round-tripped; these
        # boundary flags must be too).
        gpe_nmb.save(tmp_path / "nmb")
        gpe_mb.save(tmp_path / "mb")
        nmb_meta = yaml.safe_load((tmp_path / "nmb" / "meta.yaml").read_text())
        mb_meta = yaml.safe_load((tmp_path / "mb" / "meta.yaml").read_text())

        assert nmb_meta["merge_brackets"] is False
        assert mb_meta["merge_brackets"] is True
        assert nmb_meta["split_structure"] is True
        assert mb_meta["split_structure"] is True

    def test_round_trip_preserves_boundary_flags(
        self, gpe_mb: SmirkAdapter, tmp_path: Path
    ) -> None:
        gpe_mb.save(tmp_path)
        restored = SmirkAdapter.load(tmp_path)

        assert restored._merge_brackets == gpe_mb._merge_brackets
        assert restored._split_structure == gpe_mb._split_structure


class TestVocabStructuralIntegrity:
    """Every trained/baseline artifact has a well-formed embedding axis.

    The measurements index tokens by id and assume a dense, gap-free embedding
    axis with distinct in-range specials. (For Smirk-GPE derived merges are
    reachable only via the id→token reverse map, so contiguity is checked with
    ``id_to_token`` over ``range(len(tok))``, not the forward ``get_vocab()``.)
    """

    @pytest.mark.parametrize("tok_fixture", ["base_tok", "gpe_v200", "unigram_v200"])
    def test_every_embedding_row_resolves_to_a_token(
        self, request: pytest.FixtureRequest, tok_fixture: str
    ) -> None:
        tok = request.getfixturevalue(tok_fixture)
        for tid in range(len(tok)):
            surface = tok.id_to_token(tid)
            assert isinstance(surface, str), f"id {tid} unresolved in {tok_fixture}"
            assert surface != "", f"id {tid} resolves to empty in {tok_fixture}"

    @pytest.mark.parametrize("tok_fixture", ["base_tok", "gpe_v200", "unigram_v200"])
    def test_specials_are_distinct_and_in_range(
        self, request: pytest.FixtureRequest, tok_fixture: str
    ) -> None:
        tok = request.getfixturevalue(tok_fixture)
        special_ids = [tok.bos_id, tok.eos_id, tok.pad_id, tok.unk_id]
        assert None not in special_ids, "unk must be set so coverage's no-UNK holds"
        assert len(set(special_ids)) == len(special_ids), "special ids must be distinct"
        assert all(0 <= sid < len(tok) for sid in special_ids)

    @pytest.mark.parametrize("tok_fixture", ["base_tok", "gpe_v200", "unigram_v200"])
    def test_wordlevel_base_does_not_exceed_embedding_axis(
        self, request: pytest.FixtureRequest, tok_fixture: str
    ) -> None:
        tok = request.getfixturevalue(tok_fixture)
        assert tok.vocab_size <= len(tok)


# Golden fingerprint of the OpenSMILES glyph base (sorted surfaces, BLAKE2b-128).
# Pins the *composition* of the fixed base so a smirk-fork / rdkit bump that
# silently changed the alphabet is caught. Recompute deliberately if the base
# is intentionally changed.
_BASE_GLYPH_COUNT = 158
_BASE_GLYPH_SHA = "c563acba9cd6d4355612b49e8fcb0e77"


def _base_glyph_fingerprint(adapter: SmirkAdapter) -> str:
    glyphs = sorted(_base_glyph_tokens(adapter))
    return hashlib.blake2b("\n".join(glyphs).encode(), digest_size=16).hexdigest()


class TestBaseIsFixedAndShared:
    """The OpenSMILES base is a fixed set both arms hold identically.

    The headline Jaccard "excludes the base because both arms retain it in full"
    — valid only if the base is (a) a fixed, known set and (b) identical
    across the two arms at a matched corpus. Smirk-GPE surfaces exactly the base
    through its forward vocab (merges live in the reverse map), so its forward
    glyph set must equal the atomic baseline's; Unigram holds the same base plus
    its own multi-glyph pieces.
    """

    def test_atomic_base_has_the_fixed_size_and_composition(
        self, base_tok: SmirkAdapter
    ) -> None:
        assert base_tok.vocab_size == ATOMIC_VOCAB_SIZE
        assert len(_base_glyph_tokens(base_tok)) == _BASE_GLYPH_COUNT
        assert _base_glyph_fingerprint(base_tok) == _BASE_GLYPH_SHA

    def test_gpe_forward_base_equals_the_atomic_base_exactly(
        self, base_tok: SmirkAdapter, gpe_v200: SmirkAdapter
    ) -> None:
        assert _base_glyph_tokens(gpe_v200) == _base_glyph_tokens(base_tok)

    def test_both_arms_share_the_identical_base(
        self,
        base_tok: SmirkAdapter,
        gpe_v200: SmirkAdapter,
        unigram_v200: UnigramSmirkAdapter,
    ) -> None:
        # The set the Jaccard excludes is the same for both arms: each contains
        # the full atomic base (Unigram may also carry extra multi-glyph pieces).
        atomic_base = _base_glyph_tokens(base_tok)
        assert atomic_base <= _base_glyph_tokens(gpe_v200)
        assert atomic_base <= _base_glyph_tokens(unigram_v200)


# A broad set of canonical, OpenSMILES-conformant SMILES exercising aromatics,
# fused rings, @/@@ stereo, isotopes, charges, ionic dots, E/Z bond stereo, and
# triple bonds — each verified to round-trip exactly through the atomic base.
_CANONICAL_PROBES: tuple[str, ...] = (
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "C[C@@H](N)C(=O)O",
    "[13CH4]",
    "[NH4+]",
    "C[O-]",
    "O=S(=O)(O)O",
    "c1ccc2ccccc2c1",
    "FC(F)(F)C(=O)O",
    "C1CC2CCC1CC2",
    "ClC(Cl)(Cl)Cl",
    "[Na+].[Cl-]",
    "C/C=C/C",
    "N#Cc1ccccc1",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
)

_TOK_FIXTURES = ["base_tok", "gpe_v200", "unigram_v200"]


class TestEncodeDeterminismAndBatchAgreement:
    """Encoding is deterministic and ``encode_batch`` matches per-element ``encode``.

    Every measurement encodes the held-out split — often via the rust-parallel
    ``encode_batch`` — and assumes a stable, order-independent encoder. If batch
    encoding diverged from per-element encoding, or a repeat call drifted, the
    deposited fertility / inventory numbers would silently depend on batching.
    """

    @pytest.mark.parametrize("tok_fixture", _TOK_FIXTURES)
    def test_repeated_encode_is_identical(
        self, request: pytest.FixtureRequest, tok_fixture: str
    ) -> None:
        tok = request.getfixturevalue(tok_fixture)
        for smi in _CANONICAL_PROBES:
            assert tok.encode(smi) == tok.encode(smi)

    @pytest.mark.parametrize("tok_fixture", _TOK_FIXTURES)
    def test_batch_matches_per_element(
        self, request: pytest.FixtureRequest, tok_fixture: str
    ) -> None:
        tok = request.getfixturevalue(tok_fixture)
        batched = tok.encode_batch(list(_CANONICAL_PROBES))
        per_element = [tok.encode(smi) for smi in _CANONICAL_PROBES]
        assert batched == per_element

    @pytest.mark.parametrize("tok_fixture", _TOK_FIXTURES)
    def test_special_tokens_wrap_consistently_single_and_batch(
        self, request: pytest.FixtureRequest, tok_fixture: str
    ) -> None:
        tok = request.getfixturevalue(tok_fixture)
        bare = [tok.encode(smi) for smi in _CANONICAL_PROBES]
        wrapped = tok.encode_batch(list(_CANONICAL_PROBES), add_special_tokens=True)
        for smi, bare_ids, wrapped_ids in zip(
            _CANONICAL_PROBES, bare, wrapped, strict=True
        ):
            assert tok.encode(smi, add_special_tokens=True) == wrapped_ids
            assert wrapped_ids == [tok.bos_id, *bare_ids, tok.eos_id]

    @pytest.mark.parametrize("tok_fixture", _TOK_FIXTURES)
    def test_encode_batch_empty_list_returns_empty(
        self, request: pytest.FixtureRequest, tok_fixture: str
    ) -> None:
        # The documented empty-list short-circuit: callers (the chunked audit
        # loop) must not have to special-case a trailing empty chunk.
        tok = request.getfixturevalue(tok_fixture)
        assert tok.encode_batch([]) == []
        assert tok.encode_batch([], add_special_tokens=True) == []


class TestDecodeLosslessness:
    """Both arms losslessly round-trip canonical SMILES with no [UNK].

    The coverage guarantee at the *string* level: a tokenizer that segmented a
    molecule lossily (or emitted [UNK]) would corrupt every downstream encode of
    the held-out split. Checked across diverse structural features for the
    baseline and both trained arms.
    """

    @pytest.mark.parametrize("tok_fixture", _TOK_FIXTURES)
    def test_round_trips_exactly_without_unk(
        self, request: pytest.FixtureRequest, tok_fixture: str
    ) -> None:
        tok = request.getfixturevalue(tok_fixture)
        for smi in _CANONICAL_PROBES:
            ids = tok.encode(smi)
            assert tok.unk_id not in ids, f"{smi!r} produced [UNK] in {tok_fixture}"
            assert tok.decode(ids) == smi, f"{smi!r} broke round-trip in {tok_fixture}"


class TestMergeCountConsistency:
    """``n_merges`` agrees across the property, ``merges.txt``, ``model.merges``,
    and ``meta.yaml``.

    The scaffold measurement derives the surviving-merge id range as
    ``range(atomic, atomic + n_merges)`` from the saved ``meta.yaml`` and reads
    surfaces from ``merges.txt`` / ``model.merges``; if these counts disagreed,
    the scaffold measurement would mis-define which merges count as "in V".
    They all flow from one
    rust merge list and must stay locked together.
    """

    def test_all_four_merge_counts_match(
        self, gpe_v200: SmirkAdapter, tmp_path: Path
    ) -> None:
        gpe_v200.save(tmp_path)
        meta = yaml.safe_load((tmp_path / "meta.yaml").read_text())
        merges_txt = (tmp_path / "merges.txt").read_text().splitlines()[1:]
        model_merges = json.loads((tmp_path / "tokenizer.json").read_text())["model"][
            "merges"
        ]

        n = gpe_v200.n_merges
        assert n is not None
        assert n == len(merges_txt) == len(model_merges) == meta["n_merges"]


class TestUnigramSaveLoad:
    """The Unigram artifact contract round-trips (no merges.txt).

    ``SmirkAdapter``'s save/load is exercised throughout; its Unigram sibling
    persists a different field set (the four Unigram-LM knobs, no merge list),
    so its own round-trip is pinned here.
    """

    def test_save_emits_meta_and_no_merges(
        self, unigram_v200: UnigramSmirkAdapter, tmp_path: Path
    ) -> None:
        unigram_v200.save(tmp_path)

        assert (tmp_path / "tokenizer.json").exists()
        assert (tmp_path / "tokenizer_config.json").exists()
        assert (tmp_path / "special_tokens_map.json").exists()
        assert (tmp_path / "meta.yaml").exists()
        assert not (tmp_path / "merges.txt").exists()

    def test_round_trip_preserves_encoding_and_unigram_knobs(
        self, unigram_v200: UnigramSmirkAdapter, tmp_path: Path
    ) -> None:
        unigram_v200.save(tmp_path)
        restored = UnigramSmirkAdapter.load(tmp_path)

        assert restored.base_kind == unigram_v200.base_kind
        assert restored.name == unigram_v200.name
        assert restored.vocab_size == unigram_v200.vocab_size
        assert restored.encode(ASPIRIN) == unigram_v200.encode(ASPIRIN)
        # The four Unigram-LM knobs survive the meta.yaml round-trip.
        assert restored._seed_size == unigram_v200._seed_size
        assert restored._max_piece_length == unigram_v200._max_piece_length
        assert restored._n_sub_iterations == unigram_v200._n_sub_iterations
        assert restored._shrinking_factor == unigram_v200._shrinking_factor


class TestArtifactKindGuard:
    """``load`` rejects an artifact dir whose ``base_kind`` is the wrong arm.

    Each adapter validates ``meta.base_kind`` against its own kinds so a Smirk
    dir loaded as Unigram (or vice versa) fails loudly rather than silently
    constructing the wrong runtime wrapper.
    """

    def test_loading_gpe_dir_as_unigram_rejects(
        self, gpe_v200: SmirkAdapter, tmp_path: Path
    ) -> None:
        gpe_v200.save(tmp_path)
        with pytest.raises(ValueError, match="not a Smirk Unigram artifact"):
            UnigramSmirkAdapter.load(tmp_path)

    def test_loading_unigram_dir_as_smirk_rejects(
        self, unigram_v200: UnigramSmirkAdapter, tmp_path: Path
    ) -> None:
        unigram_v200.save(tmp_path)
        with pytest.raises(ValueError, match="not a Smirk artifact"):
            SmirkAdapter.load(tmp_path)


class TestSmirkAdapterInitGuard:
    """``SmirkAdapter`` rejects a base_kind outside its two kinds."""

    def test_init_rejects_unknown_base_kind(self) -> None:
        with pytest.raises(ValueError, match="base_kind must be"):
            SmirkAdapter(
                name="bad",
                base_kind="not_a_kind",  # type: ignore[arg-type]
                tokenizer=SmirkTokenizerFast(),
            )


class TestScaffoldLogMaterialization:
    """`_materialize_scaffold_log` honors the self-contained-artifact contract.

    When the scaffold log was written outside the artifact dir, ``save`` copies
    it in so the on-disk artifact is self-contained; when the recorded log path
    no longer exists, ``save`` degrades to recording no sha rather than failing.
    """

    def test_log_copied_in_when_written_outside_artifact_dir(
        self, synth_corpus_path: Path, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "logs" / "scaffold.jsonl"
        log_path.parent.mkdir()
        tok = SmirkAdapter.train_gpe(
            [synth_corpus_path],
            name="copied",
            vocab_size=200,
            min_frequency=2,
            scaffold_log_path=log_path,
        )
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        tok.save(artifact_dir)

        copied = artifact_dir / "scaffold.jsonl"
        assert copied.is_file()
        assert copied.read_bytes() == log_path.read_bytes()
        meta = yaml.safe_load((artifact_dir / "meta.yaml").read_text())
        assert (
            meta["scaffold_log_sha"] == hashlib.sha256(copied.read_bytes()).hexdigest()
        )

    def test_missing_log_records_no_sha(
        self, base_tok: SmirkAdapter, tmp_path: Path
    ) -> None:
        # base_tok is function-scoped, so mutating its log path is local to
        # this test. Point it at a file that was never written.
        base_tok._scaffold_log_path = tmp_path / "never_written.jsonl"
        base_tok.save(tmp_path)

        meta = yaml.safe_load((tmp_path / "meta.yaml").read_text())
        assert meta["scaffold_log_sha"] is None
        assert not (tmp_path / "scaffold.jsonl").exists()
