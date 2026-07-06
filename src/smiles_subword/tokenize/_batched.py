"""Batched-encode helper for the audit hot path: chunk an ``Iterable[str]``,
encode each chunk via the rust-side parallel path, yield one id-list per
molecule in input order (avoids per-SMILES FFI cost + engages the rayon pool).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from smiles_subword.tokenize.base import Tokenizer


ENCODE_BATCH_SIZE = 8192
"""SMILES per ``encode_batch`` call. 8K: engages the rayon pool on an
8-core box while peak memory stays under a few hundred MB.
"""


def iter_encoded_batches(
    tok: Tokenizer,
    smiles: Iterable[str],
    *,
    add_special_tokens: bool = False,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> Iterator[list[int]]:
    """Yield one id-list per SMILES, encoding in ``batch_size`` chunks.

    Args:
        tok: tokenizer satisfying the project ``Tokenizer`` Protocol.
        smiles: lazy iterator over SMILES strings; consumed once.
        add_special_tokens: forwarded to ``encode_batch``; audit convention
            strips specials in the metric layer, not the tokenizer.
        batch_size: SMILES per ``encode_batch`` call. Override only for tests
            or memory-constrained debug runs.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size!r}")
    chunk: list[str] = []
    for smi in smiles:
        chunk.append(smi)
        if len(chunk) >= batch_size:
            yield from tok.encode_batch(chunk, add_special_tokens=add_special_tokens)
            chunk = []
    if chunk:
        yield from tok.encode_batch(chunk, add_special_tokens=add_special_tokens)
