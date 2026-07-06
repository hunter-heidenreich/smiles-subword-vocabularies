"""Tests for ``measure._tables`` (``*_table.md`` cell formatting).

Pins ``fmt_md`` directly, including the None / NaN branches the table-rendering
paths don't otherwise reach.
"""

from __future__ import annotations

from smiles_subword.tokenize.measure._tables import fmt_md


class TestFmtMd:
    def test_none_renders_em_dash(self) -> None:
        assert fmt_md(None) == "—"

    def test_nan_renders_nan(self) -> None:
        assert fmt_md(float("nan")) == "nan"

    def test_default_is_four_decimals(self) -> None:
        assert fmt_md(0.5) == "0.5000"
        assert fmt_md(1 / 3) == "0.3333"

    def test_signed_spec_forces_a_sign(self) -> None:
        assert fmt_md(0.5, spec="+.4f") == "+0.5000"
        assert fmt_md(-0.5, spec="+.4f") == "-0.5000"

    def test_custom_spec_is_honored(self) -> None:
        assert fmt_md(0.5, spec=".2f") == "0.50"

    def test_integer_value_is_coerced_to_float(self) -> None:
        assert fmt_md(2) == "2.0000"
