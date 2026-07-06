"""Validity tests for the results-table extraction layer (``results/build/extract``).

This is the last mile of the measurement chain: ``extract`` selects which
deposited number lands in which published table cell, and in what order. A bug
here renders a clean, well-formatted, *wrong* table that the author will not
catch by eye. These tests pin the selection/derivation logic — narrative row
order, CI parsing, grid-vs-extras gating, the deposit-field -> published-field
transcription (especially the renamed fields), and the cross-table joins — such
that a plausible mis-wiring (swapped field, wrong sibling cell, dropped row)
fails rather than silently shipping.

The builders read deposits off disk via ``read_table`` / ``read_audit`` /
``read_deadzone_cell``; we monkeypatch those to feed crafted payloads, so every
golden ties a known deposit value to the row field it must populate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import extract
import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping


# --------------------------------------------------------------------------- #
# Deposit-payload fixtures                                                     #
# --------------------------------------------------------------------------- #


def _patch_tables(monkeypatch: pytest.MonkeyPatch, tables: Mapping[str, Any]) -> None:
    """Route ``read_table(name)`` to ``tables[name]`` (``None`` when absent)."""
    monkeypatch.setattr(extract, "read_table", tables.get)


def _matched(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"matched": rows}


def _coord(
    corpus: str = "pubchem",
    vocab_size: int = 256,
    boundary: str = "nmb",
    *,
    tier: str = "headline",
    extras_kind: object = None,
) -> dict[str, Any]:
    """The common coordinate/identity columns every matched row carries."""
    return {
        "pair_key": f"{corpus}__{vocab_size}_{boundary}",
        "corpus": corpus,
        "vocab_size": vocab_size,
        "boundary": boundary,
        "tier": tier,
        "extras_kind": extras_kind,
    }


# --------------------------------------------------------------------------- #
# Narrative row ordering                                                       #
# --------------------------------------------------------------------------- #


class TestCoordSortKey:
    """Rows must read in the paper's narrative order, whatever the deposit order."""

    def test_corpus_typology_then_v_then_boundary(self) -> None:
        # PubChem -> ZINC-22 -> COCONUT -> REAL-Space; V ascending; NMB before MB.
        assert extract._coord_sort_key("pubchem", 256, "nmb") < extract._coord_sort_key(
            "zinc22", 256, "nmb"
        )
        assert extract._coord_sort_key("zinc22", 256, "nmb") < extract._coord_sort_key(
            "coconut", 256, "nmb"
        )
        assert extract._coord_sort_key("coconut", 256, "nmb") < extract._coord_sort_key(
            "real_space", 256, "nmb"
        )
        # V ascending dominates boundary within a corpus.
        assert extract._coord_sort_key("pubchem", 256, "mb") < extract._coord_sort_key(
            "pubchem", 512, "nmb"
        )
        # NMB before MB at the same (corpus, V).
        assert extract._coord_sort_key("pubchem", 256, "nmb") < extract._coord_sort_key(
            "pubchem", 256, "mb"
        )

    def test_unknown_corpus_and_boundary_sort_last(self) -> None:
        known = extract._coord_sort_key("real_space", 256, "nmb")
        unknown_corpus = extract._coord_sort_key("ood_corpus", 256, "nmb")
        unknown_boundary = extract._coord_sort_key("pubchem", 256, "weird")
        assert unknown_corpus > known  # rank 99 fallback
        assert unknown_boundary[2] == 99  # boundary-order fallback

    def test_builder_emits_rows_in_narrative_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Hand the builder a deliberately scrambled deposit; demand sorted output.
        scrambled = [
            {**_coord("ood", 256, "nmb"), "jaccard": 0.1},  # unknown -> last
            {**_coord("pubchem", 512, "nmb"), "jaccard": 0.1},
            {**_coord("coconut", 256, "nmb"), "jaccard": 0.1},
            {**_coord("pubchem", 256, "mb"), "jaccard": 0.1},
            {**_coord("zinc22", 256, "nmb"), "jaccard": 0.1},
            {**_coord("pubchem", 256, "nmb"), "jaccard": 0.1},
            {**_coord("real_space", 256, "nmb"), "jaccard": 0.1},
        ]
        _patch_tables(monkeypatch, {extract.JACCARD_TABLE: _matched(scrambled)})

        keys = [r.pair_key for r in extract.jaccard_rows()]
        assert keys == [
            "pubchem__256_nmb",
            "pubchem__256_mb",
            "pubchem__512_nmb",
            "zinc22__256_nmb",
            "coconut__256_nmb",
            "real_space__256_nmb",
            "ood__256_nmb",
        ]


# --------------------------------------------------------------------------- #
# CI parsing                                                                   #
# --------------------------------------------------------------------------- #


class TestRowCi:
    """A CI is reported only when *both* bounds are present; never half a bar."""

    def test_both_bounds_present(self) -> None:
        row = {"x_ci_lo": 0.1, "x_ci_hi": 0.3}
        assert extract._row_ci(row, "x") == (0.1, 0.3)

    def test_missing_either_bound_yields_none(self) -> None:
        assert extract._row_ci({"x_ci_lo": 0.1}, "x") is None  # hi missing
        assert extract._row_ci({"x_ci_hi": 0.3}, "x") is None  # lo missing
        assert extract._row_ci({}, "x") is None  # both missing

    def test_prefix_is_respected(self) -> None:
        # The prefix selects the bar; a sibling prefix must not leak in.
        row = {"a_ci_lo": 0.1, "a_ci_hi": 0.2, "b_ci_lo": 0.8, "b_ci_hi": 0.9}
        assert extract._row_ci(row, "a") == (0.1, 0.2)
        assert extract._row_ci(row, "b") == (0.8, 0.9)


# --------------------------------------------------------------------------- #
# Grid-vs-extras gating                                                        #
# --------------------------------------------------------------------------- #


class TestGridGating:
    """``extras_kind is None`` is the grid; extras enter only when asked for."""

    def test_is_grid_only_when_extras_kind_none(self) -> None:
        assert extract._is_grid({"extras_kind": None}) is True
        assert extract._is_grid({}) is True  # absent key == grid
        assert extract._is_grid({"extras_kind": "subsample_redraw"}) is False

    def test_keep_drops_extras_by_default_keeps_with_flag(self) -> None:
        grid = {"extras_kind": None}
        extra = {"extras_kind": "seed_cap"}
        assert extract._keep(grid, include_extras=False) is True
        assert extract._keep(extra, include_extras=False) is False
        assert extract._keep(extra, include_extras=True) is True
        assert extract._keep(grid, include_extras=True) is True

    def test_builder_excludes_extras_unless_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {**_coord("pubchem", 256, "nmb"), "jaccard": 0.1},
            {
                **_coord("pubchem", 256, "nmb", extras_kind="subsample_redraw"),
                "pair_key": "pubchem__v256_nmb_r1",
                "jaccard": 0.2,
            },
        ]
        _patch_tables(monkeypatch, {extract.JACCARD_TABLE: _matched(rows)})

        default = extract.jaccard_rows()
        assert [r.pair_key for r in default] == ["pubchem__256_nmb"]

        with_extras = extract.jaccard_rows(include_extras=True)
        assert len(with_extras) == 2


# --------------------------------------------------------------------------- #
# Optional coercion helpers                                                    #
# --------------------------------------------------------------------------- #


class TestOptionalCoercion:
    def test_none_passes_through(self) -> None:
        assert extract._opt_float(None) is None
        assert extract._opt_int(None) is None
        assert extract._opt_bool(None) is None

    def test_values_are_coerced(self) -> None:
        assert extract._opt_float(1) == 1.0
        assert isinstance(extract._opt_float(1), float)
        assert extract._opt_int(2.9) == 2  # truncates like int()
        assert extract._opt_bool(0) is False
        assert extract._opt_bool(1) is True


# --------------------------------------------------------------------------- #
# Section / index defensive reads                                             #
# --------------------------------------------------------------------------- #


class TestSectionAndIndex:
    def test_section_handles_missing_table_and_section(self) -> None:
        assert extract._section(None, "matched") == []
        assert extract._section({}, "matched") == []
        assert extract._section({"matched": "not-a-list"}, "matched") == []
        assert extract._section({"matched": [{"a": 1}]}, "matched") == [{"a": 1}]

    def test_index_keys_by_pair_key(self) -> None:
        idx = extract._index([{"pair_key": "k1", "v": 1}, {"pair_key": "k2", "v": 2}])
        assert idx["k1"]["v"] == 1
        assert idx["k2"]["v"] == 2


# --------------------------------------------------------------------------- #
# Small parsing/derivation helpers                                            #
# --------------------------------------------------------------------------- #


class TestParsingHelpers:
    def test_redraw_index_reads_trailing_r_suffix(self) -> None:
        assert extract._redraw_index("pubchem__v512_nmb_r0") == 0
        assert extract._redraw_index("pubchem__v512_nmb_r12") == 12

    def test_vocab_from_cell_finds_the_v_token(self) -> None:
        assert extract._vocab_from_cell("pubchem__smirk_unigram_v1024_nmb") == 1024
        assert extract._vocab_from_cell("zinc22_v256_mb") == 256
        # A non-numeric "v..." token must not be mistaken for a vocab size.
        assert extract._vocab_from_cell("version_only_no_size") == 0

    def test_redraw_spread_is_max_minus_min(self) -> None:
        spread = extract.RedrawSpread("pubchem", (0.90, 0.95, 0.88)).spread
        assert spread == pytest.approx(0.07)

    def test_redraw_spread_degenerate_cases(self) -> None:
        assert extract.RedrawSpread("pubchem", (0.9,)).spread == 0.0  # single -> 0
        assert extract.RedrawSpread("pubchem", ()).spread == 0.0  # empty -> 0


# --------------------------------------------------------------------------- #
# Content SHA                                                                  #
# --------------------------------------------------------------------------- #


class TestPayloadSha:
    def test_is_order_independent_but_content_sensitive(self) -> None:
        a = extract._payload_sha({"x": 1, "y": 2})
        b = extract._payload_sha({"y": 2, "x": 1})  # same content, keys reordered
        c = extract._payload_sha({"x": 1, "y": 3})  # one value changed
        assert a == b  # canonicalized -> key order irrelevant
        assert a != c  # content change must move the SHA

    def test_upstream_sha_map_marks_absent_tables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        present = {"matched": []}
        # Only one required table present; the rest must read "absent".
        _patch_tables(monkeypatch, {extract.JACCARD_TABLE: present})
        shas = extract.upstream_sha_map()
        assert shas[extract.JACCARD_TABLE] != "absent"
        assert shas[extract.DEADZONE_TABLE] == "absent"
        assert set(shas) == set(extract.REQUIRED_TABLES)


# --------------------------------------------------------------------------- #
# Deposit-field -> published-field transcription (the rename traps)            #
# --------------------------------------------------------------------------- #


class TestTranscription:
    """Each builder must map the *named* deposit key to the *named* row field.

    Renamed fields are the trap: ``*_fraction`` -> ``*_absorbed``,
    ``*_bag_explicitH`` -> ``*_explicit_h``, ``delta_fertility_relative`` ->
    ``rel_fertility``. Distinct golden values per field catch a swap.
    """

    def test_jaccard_row_maps_three_jaccards_and_ci(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = {
            **_coord(),
            "jaccard": 0.33,
            "jaccard_struct": 0.44,
            "weighted_jaccard": 0.55,
            "weighted_jaccard_ci_lo": 0.50,
            "weighted_jaccard_ci_hi": 0.60,
            "weighted_jaccard_struct": 0.66,
            "weighted_jaccard_struct_ci_lo": 0.61,
            "weighted_jaccard_struct_ci_hi": 0.71,
            "bpe_n_multi": 90,
            "unigram_n_multi": 91,
        }
        _patch_tables(monkeypatch, {extract.JACCARD_TABLE: _matched([row])})
        (jr,) = extract.jaccard_rows()
        assert jr.jaccard == 0.33
        assert jr.jaccard_struct == 0.44
        assert jr.weighted_jaccard == 0.55
        assert jr.weighted_jaccard_ci == (0.50, 0.60)
        assert jr.weighted_jaccard_struct == 0.66
        assert jr.weighted_jaccard_struct_ci == (0.61, 0.71)
        assert jr.bpe_n_multi == 90
        assert jr.unigram_n_multi == 91

    def test_jaccard_struct_ci_independent_of_weighted_ci(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If only the plain weighted CI is present, the struct CI must stay None
        # (and vice versa) — the two bars must not cross-contaminate.
        row = {
            **_coord(),
            "jaccard": 0.1,
            "weighted_jaccard_ci_lo": 0.4,
            "weighted_jaccard_ci_hi": 0.5,
        }
        _patch_tables(monkeypatch, {extract.JACCARD_TABLE: _matched([row])})
        (jr,) = extract.jaccard_rows()
        assert jr.weighted_jaccard_ci == (0.4, 0.5)
        assert jr.weighted_jaccard_struct_ci is None

    def test_fertility_row_renames_relative_to_rel_fertility(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = {
            **_coord(),
            "bpe_fertility": 10.0,
            "bpe_fertility_ci_lo": 9.5,
            "bpe_fertility_ci_hi": 10.5,
            "unigram_fertility": 12.0,
            "unigram_fertility_ci_lo": 11.5,
            "unigram_fertility_ci_hi": 12.5,
            "bpe_glyphs_per_token": 2.0,
            "bpe_glyphs_per_token_ci_lo": 1.9,
            "bpe_glyphs_per_token_ci_hi": 2.1,
            "unigram_glyphs_per_token": 2.4,
            "unigram_glyphs_per_token_ci_lo": 2.3,
            "unigram_glyphs_per_token_ci_hi": 2.5,
            "delta_fertility": 2.0,
            "delta_fertility_relative": 0.2,
        }
        _patch_tables(monkeypatch, {extract.FERTILITY_TABLE: _matched([row])})
        (fr,) = extract.fertility_rows()
        assert fr.bpe_fertility == 10.0
        assert fr.bpe_fertility_ci == (9.5, 10.5)
        assert fr.unigram_fertility == 12.0
        assert fr.bpe_glyphs_per_token == 2.0
        assert fr.unigram_glyphs_per_token == 2.4
        assert fr.delta_fertility == 2.0  # absolute gap
        assert fr.rel_fertility == 0.2  # <- delta_fertility_relative, renamed

    def test_absorption_row_point_from_fraction_ci_from_bare_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The point lives under ``*_absorbed_fraction`` but the CI under the bare
        # ``*_absorbed`` prefix — pin both halves of that asymmetry.
        row = {
            **_coord(),
            "bpe_absorbed_fraction": 0.30,
            "bpe_absorbed_ci_lo": 0.28,
            "bpe_absorbed_ci_hi": 0.32,
            "unigram_absorbed_fraction": 0.50,
            "unigram_absorbed_ci_lo": 0.48,
            "unigram_absorbed_ci_hi": 0.52,
            "delta_absorbed": 0.20,
        }
        _patch_tables(monkeypatch, {extract.ABSORPTION_TABLE: _matched([row])})
        (ar,) = extract.absorption_rows()
        assert ar.bpe_absorbed == 0.30
        assert ar.bpe_absorbed_ci == (0.28, 0.32)
        assert ar.unigram_absorbed == 0.50
        assert ar.unigram_absorbed_ci == (0.48, 0.52)
        assert ar.delta_absorbed == 0.20

    def test_noncanon_row_renames_bag_axes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``*_bag_<axis>`` -> ``*_<axis>``; ``explicitH`` -> ``explicit_h``.
        row = {
            **_coord(),
            "bpe_bag_random": 0.11,
            "ul_bag_random": 0.12,
            "bpe_bag_kekule": 0.21,
            "ul_bag_kekule": 0.22,
            "bpe_bag_explicitH": 0.31,
            "ul_bag_explicitH": 0.32,
            "bpe_bag_obcanon": 0.41,
            "ul_bag_obcanon": 0.42,
            "gap_canon": 1.4,
            "gap_rand": 1.3,
        }
        _patch_tables(monkeypatch, {extract.NONCANON_TABLE: _matched([row])})
        (nr,) = extract.noncanon_rows()
        assert (nr.bpe_random, nr.ul_random) == (0.11, 0.12)
        assert (nr.bpe_kekule, nr.ul_kekule) == (0.21, 0.22)
        assert (nr.bpe_explicit_h, nr.ul_explicit_h) == (0.31, 0.32)
        assert (nr.bpe_obcanon, nr.ul_obcanon) == (0.41, 0.42)
        # gap_canon/gap_rand are deposited as Unigram/BPE fertility *ratios* and
        # converted to the paper's symmetric relative gap rel|df| = 2(r-1)/(r+1).
        assert nr.gap_canon == pytest.approx(2 * (1.4 - 1) / (1.4 + 1))
        assert nr.gap_rand == pytest.approx(2 * (1.3 - 1) / (1.3 + 1))

    def test_delta_f_row_carries_clearances_and_unsafe_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = {
            **_coord(),
            "bpe_headline_clearance": 0.97,
            "unigram_headline_clearance": 0.80,
            "headline_delta_f": 0.17,
            "any_arm_unsafe": True,
        }
        _patch_tables(monkeypatch, {extract.DEADZONE_TABLE: _matched([row])})
        (dr,) = extract.delta_f_rows()
        assert dr.bpe_clearance == 0.97
        assert dr.unigram_clearance == 0.80
        assert dr.headline_delta_f == 0.17
        assert dr.any_arm_unsafe is True


# --------------------------------------------------------------------------- #
# Deadzone n-sweep: reshape + per-condition deposit join                       #
# --------------------------------------------------------------------------- #


class TestDeadzoneNSweep:
    def test_clearance_by_n_keys_coerced_to_int(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spine = [{**_coord("pubchem", 256, "nmb")}]
        _patch_tables(monkeypatch, {extract.DEADZONE_TABLE: _matched(spine)})
        monkeypatch.setattr(
            extract,
            "read_deadzone_cell",
            lambda _key: {
                "bpe": {"clearance_by_n": {"1": 0.99, "100": 0.95}},
                "unigram": {"clearance_by_n": {"1": 0.90, "100": 0.70}},
            },
        )
        (row,) = extract.deadzone_nsweep_rows()
        assert row.bpe_c == {1: 0.99, 100: 0.95}  # str JSON keys -> int
        assert row.unigram_c == {1: 0.90, 100: 0.70}

    def test_row_dropped_when_per_condition_deposit_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spine = [{**_coord("pubchem", 256, "nmb")}]
        _patch_tables(monkeypatch, {extract.DEADZONE_TABLE: _matched(spine)})
        monkeypatch.setattr(extract, "read_deadzone_cell", lambda _key: None)
        assert extract.deadzone_nsweep_rows() == []

    def test_missing_arm_block_yields_empty_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spine = [{**_coord("pubchem", 256, "nmb")}]
        _patch_tables(monkeypatch, {extract.DEADZONE_TABLE: _matched(spine)})
        monkeypatch.setattr(
            extract,
            "read_deadzone_cell",
            lambda _key: {"bpe": {"clearance_by_n": {"1": 0.5}}},  # no unigram block
        )
        (row,) = extract.deadzone_nsweep_rows()
        assert row.bpe_c == {1: 0.5}
        assert row.unigram_c == {}  # absent arm -> empty, not a crash


# --------------------------------------------------------------------------- #
# Cross-table joins                                                            #
# --------------------------------------------------------------------------- #


def _join_tables(key: str = "pubchem__256_nmb") -> dict[str, Any]:
    """A consistent set of matched deposits sharing one ``pair_key``."""
    coord = {**_coord("pubchem", 256, "nmb")}
    coord["pair_key"] = key
    return {
        extract.DEADZONE_TABLE: _matched([{**coord, "headline_delta_f": 0.17}]),
        extract.JACCARD_TABLE: _matched([{**coord, "jaccard": 0.30}]),
        extract.FERTILITY_TABLE: _matched(
            [
                {
                    **coord,
                    "bpe_fertility": 10.0,
                    "unigram_fertility": 12.0,
                    "bpe_glyphs_per_token": 2.0,
                    "unigram_glyphs_per_token": 2.4,
                    "delta_fertility": 2.0,
                    "delta_fertility_relative": 0.20,
                }
            ]
        ),
        extract.DISTRIBUTION_TABLE: _matched(
            [
                {
                    **coord,
                    "abs_delta_d": 0.05,
                    "delta_d": -0.05,
                    "bpe_d": 0.1,
                    "unigram_d": 0.15,
                    "bpe_eta": 0.9,
                    "unigram_eta": 0.8,
                    "bpe_renyi": 0.7,
                    "unigram_renyi": 0.6,
                }
            ]
        ),
        extract.ABSORPTION_TABLE: _matched([{**coord, "delta_absorbed": 0.20}]),
        extract.SCAFFOLD_TABLE: _matched([{**coord, "delta_scaffold_fraction": 0.03}]),
        extract.SEGMENTATION_TABLE: _matched(
            [{**coord, "delta_entropy_per_glyph": 0.08}]
        ),
    }


class TestMeasurementRowsJoin:
    """The seven-scalar spine join: Deadzone spine ⋈ J/fert/dist (required) +
    absorption/scaffold/segmentation (optional -> ``None`` when absent)."""

    def test_full_join_pulls_each_scalar_from_its_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_tables(monkeypatch, _join_tables())
        (mr,) = extract.measurement_rows()
        assert mr.delta_f == 0.17  # deadzone
        assert mr.jaccard == 0.30  # jaccard
        assert mr.rel_fertility == 0.20  # fertility (relative)
        assert mr.abs_delta_d == 0.05  # distribution
        assert mr.delta_absorbed == 0.20  # absorption
        assert mr.delta_scaffold == 0.03  # scaffold (_fraction key)
        assert mr.delta_entropy_per_glyph == 0.08  # segmentation

    def test_optional_measures_become_none_when_their_table_lacks_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tables = _join_tables()
        tables[extract.ABSORPTION_TABLE] = _matched([])
        tables[extract.SCAFFOLD_TABLE] = _matched([])
        tables[extract.SEGMENTATION_TABLE] = _matched([])
        _patch_tables(monkeypatch, tables)
        (mr,) = extract.measurement_rows()
        assert mr.delta_absorbed is None
        assert mr.delta_scaffold is None
        assert mr.delta_entropy_per_glyph is None
        # The required spine scalars are still present.
        assert mr.jaccard == 0.30
        assert mr.rel_fertility == 0.20

    def test_missing_required_join_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Jaccard is a required spine join; a deadzone cell with no Jaccard
        # counterpart is a corrupt deposit set, not a silently-dropped row.
        tables = _join_tables()
        tables[extract.JACCARD_TABLE] = _matched([])
        _patch_tables(monkeypatch, tables)
        with pytest.raises(KeyError):
            extract.measurement_rows()


class TestCrossAxisCellsJoin:
    """``cross_axis_cells`` joins J/fert/dist but *skips* a cell whose fertility
    or distribution counterpart is missing (unlike the measurement spine, which
    raises) — the figures simply omit an incomplete cell."""

    def test_full_join(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_tables(monkeypatch, _join_tables())
        (cell,) = extract.cross_axis_cells()
        assert cell.jaccard == 0.30
        assert cell.rel_fertility == 0.20
        assert cell.abs_delta_d == 0.05
        assert cell.delta_fertility_signed == 2.0
        assert cell.delta_d_signed == -0.05

    def test_cell_skipped_when_a_join_partner_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tables = _join_tables()
        tables[extract.FERTILITY_TABLE] = _matched([])  # no fertility for the cell
        _patch_tables(monkeypatch, tables)
        assert extract.cross_axis_cells() == []  # skipped, not raised


# --------------------------------------------------------------------------- #
# Robustness-extras assembly                                                   #
# --------------------------------------------------------------------------- #


class TestRobustnessExtras:
    def test_redraws_grouped_by_corpus_and_ordered_by_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {
                **_coord("pubchem", 512, "nmb", extras_kind="subsample_redraw"),
                "pair_key": "pubchem__v512_nmb_r2",
                "unigram_headline_clearance": 0.80,
            },
            {
                **_coord("pubchem", 512, "nmb", extras_kind="subsample_redraw"),
                "pair_key": "pubchem__v512_nmb_r0",
                "unigram_headline_clearance": 0.90,
            },
            {
                **_coord("pubchem", 512, "nmb", extras_kind="subsample_redraw"),
                "pair_key": "pubchem__v512_nmb_r1",
                "unigram_headline_clearance": 0.85,
            },
        ]
        _patch_tables(monkeypatch, {extract.DEADZONE_TABLE: _matched(rows)})
        monkeypatch.setattr(extract, "read_audit", lambda _name: None)
        ex = extract.robustness_extras()
        (rd,) = ex.redraws
        assert rd.corpus == "pubchem"
        # Ordered r0, r1, r2 by parsed index — and spread is max-min over them.
        assert rd.clearances == (0.90, 0.85, 0.80)
        assert rd.spread == pytest.approx(0.10)

    def test_size_sweep_selects_the_three_anchor_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {
                **_coord("pubchem", 512, "nmb", extras_kind="training_corpus_size"),
                "pair_key": "pubchem__v512_nmb__size_5m",
                "unigram_headline_clearance": 0.70,
            },
            {
                **_coord("pubchem", 512, "nmb", extras_kind="training_corpus_size"),
                "pair_key": "pubchem__v512_nmb__size_15m",
                "unigram_headline_clearance": 0.80,
            },
            {
                **_coord("pubchem", 512, "nmb"),
                "pair_key": "pubchem__v512_nmb",  # the 50M baseline
                "unigram_headline_clearance": 0.90,
            },
        ]
        _patch_tables(monkeypatch, {extract.DEADZONE_TABLE: _matched(rows)})
        monkeypatch.setattr(extract, "read_audit", lambda _name: None)
        ex = extract.robustness_extras()
        assert [(p.n_label, p.unigram_clearance) for p in ex.size_sweep] == [
            ("5M", 0.70),
            ("15M", 0.80),
            ("50M", 0.90),
        ]

    def test_audit_fields_read_from_their_deposits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_tables(monkeypatch, {extract.DEADZONE_TABLE: _matched([])})
        audits = {
            "seed_cap": {
                "multi_glyph_jaccard": 0.97,
                "symmetric_difference_count": 12,
            },
            "prune_schedule": {
                "comparisons": [
                    {
                        "baseline_cell": "pubchem__smirk_unigram_v1024_nmb",
                        "multi_glyph_jaccard": 0.95,
                    }
                ]
            },
            "merge_exhaustion": {
                "vocab_size_realised": 900,
                "vocab_size_cap": 1024,
                "natural_termination": True,
            },
        }
        monkeypatch.setattr(extract, "read_audit", audits.get)
        ex = extract.robustness_extras()
        assert ex.seed_cap_jaccard == 0.97
        assert ex.seed_cap_symmetric_difference == 12
        assert ex.prune == (extract.PruneComparison(vocab_size=1024, jaccard=0.95),)
        assert ex.merge_exhaustion_realised_v == 900
        assert ex.merge_exhaustion_cap == 1024
        assert ex.merge_exhaustion_natural is True

    def test_absent_audits_yield_none_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_tables(monkeypatch, {extract.DEADZONE_TABLE: _matched([])})
        monkeypatch.setattr(extract, "read_audit", lambda _name: None)
        ex = extract.robustness_extras()
        assert ex.seed_cap_jaccard is None
        assert ex.seed_cap_symmetric_difference is None
        assert ex.prune == ()
        assert ex.merge_exhaustion_realised_v is None
        assert ex.merge_exhaustion_natural is None


# --------------------------------------------------------------------------- #
# Missing-input reporting                                                      #
# --------------------------------------------------------------------------- #


class TestMissingReporting:
    def test_missing_tables_lists_absent_required_tables_sorted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_tables(monkeypatch, {extract.JACCARD_TABLE: {"matched": []}})
        missing = extract.missing_tables()
        assert extract.JACCARD_TABLE not in missing
        assert extract.DEADZONE_TABLE in missing
        assert missing == sorted(missing)
        assert set(missing) == set(extract.REQUIRED_TABLES) - {extract.JACCARD_TABLE}

    def test_missing_audits_lists_absent_audit_items_sorted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            extract,
            "read_audit",
            lambda name: {"x": 1} if name == "seed_cap" else None,
        )
        missing = extract.missing_audits()
        assert "seed_cap" not in missing
        assert "prune_schedule" in missing
        assert "merge_exhaustion" in missing
        assert missing == sorted(missing)
