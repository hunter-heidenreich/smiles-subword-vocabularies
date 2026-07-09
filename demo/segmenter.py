"""Pure-Python re-implementation of the smirk SMILES segmenter.

This module reproduces, from a saved ``tokenizer.json`` alone, the exact token
sequence that ``smirk.SmirkTokenizerFast`` (the pinned Rust fork) produces -
with **no smirk / Rust / torch dependency**, only the standard library. That is
what lets the interactive demo run on a free CPU Space without building the fork.

Validated against smirk's ``encode`` over tens of thousands of held-out SMILES
per cell (see ``validate.py``): BPE (``smirk_gpe``) is byte-for-byte exact;
Unigram matches on ≈99.99%, diverging only on rare bit-identical Viterbi score
ties in long homopolymer chains (documented on ``_viterbi``). Do not "improve"
the algorithm without re-running that check.

Pipeline (mirrors smirk's ``tokenizer.json``):

1. **Normalize** - the ``Sequence`` normalizer: ``++``->``+2``, ``--``->``-2``,
   then strip. (Read verbatim from the JSON, not hard-coded.)
2. **Glyph split** - the OpenSMILES glyph base. The ``model.tokenize.outer``
   regex matches every top-level glyph, including a whole bracket atom
   ``[...]``; ``model.tokenize.inner`` decomposes a bracket atom into its glyph
   pieces (element, chirality, H-count, charge, map-num) framed by literal
   ``[`` / ``]``.
3. **Isolation** - ``split_structure`` isolates the structural glyphs
   ``. ( ) / \\ % <digit>`` and each bracket atom into their own pretokens;
   merges / Viterbi never cross those boundaries. This equals smirk's ``Split``
   pre-tokenizer.
4. **Model** - within each mergeable run: BPE applies ranked merges greedily
   (``smirk_gpe``); Unigram-LM runs max-score Viterbi (``smirk_unigram``).

The ``merge_brackets`` axis (from ``meta.yaml``): when false (NMB) the ``[`` and
``]`` glyphs are themselves barriers, so a bracket atom stays decomposed (its
*inner* glyphs may still merge among themselves, e.g. ``[C@@H]`` -> ``[ C@@H ]``);
when true (MB) the whole bracket atom is one mergeable run (``[nH]`` -> ``[nH]``).
"""

from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Piece:
    """One output token."""

    text: str
    start: int  # char offset into the normalized SMILES (inclusive)
    end: int  # char offset (exclusive)
    n_glyphs: int  # how many glyph-base units it spans (1 == atomic glyph)

    @property
    def is_multiglyph(self) -> bool:
        return self.n_glyphs > 1


@dataclass(frozen=True)
class _Glyph:
    text: str
    start: int
    end: int


class Segmenter:
    """Reproduces ``smirk`` segmentation for a single trained cell."""

    def __init__(self, spec: dict, *, merge_brackets: bool) -> None:
        self.merge_brackets = merge_brackets
        self._replacements: list[tuple[str, str]] = []
        self._strip = False
        for n in spec.get("normalizer", {}).get("normalizers", []):
            if n["type"] == "Replace":
                self._replacements.append((n["pattern"]["String"], n["content"]))
            elif n["type"] == "Strip":
                self._strip = True

        tk = spec["model"]["tokenize"]
        self._outer = re.compile(tk["outer"])
        self._inner = re.compile(tk["inner"])
        # The pre-tokenizer's Split regex is the authority on isolation: it cuts
        # the SMILES into pretokens (delimiters `%dd`, `[...]`, `. ( ) / \`, a
        # bare digit - and the organic runs between them). Merges/Viterbi run
        # strictly inside one pretoken; they never cross these cuts.
        self._pre = re.compile(spec["pre_tokenizer"]["pattern"]["Regex"])

        model = spec["model"]
        # The multi-glyph vocabulary: every learned piece spanning >1 glyph. This
        # is the set the two arms are compared on (single-glyph atoms are shared
        # by construction, so they carry no signal about how the algorithms
        # differ). Drives the demo's arm-exclusivity colouring.
        self.multiglyph_pieces: set[str]
        if "merges" in model:
            self.algorithm = "bpe"
            self._rank, self.multiglyph_pieces = _build_merge_ranks(
                model["merges"], model["vocab"]
            )
            self._scores: dict[str, float] = {}
        else:
            self.algorithm = "unigram"
            self._rank = {}
            self._scores = {
                "".join(e["glyphs"]): float(e["score"]) for e in model["vocab"]
            }
            # Derived exactly as smirk's UnigramModel::new does (not serialized):
            #   max_piece_len bounds the Viterbi span; unk_score is charged for a
            #   single OOV glyph and equals the min piece score (0.0 if none).
            self._max_piece_len = max(
                (len(e["glyphs"]) for e in model["vocab"]), default=1
            )
            scores = [float(e["score"]) for e in model["vocab"]]
            self._unk_score = min(scores) if scores else 0.0
            self._unk_token = model.get("unk_token", "[UNK]")
            self.multiglyph_pieces = {
                "".join(e["glyphs"]) for e in model["vocab"] if len(e["glyphs"]) > 1
            }

    # -- construction --------------------------------------------------------
    @classmethod
    def from_dir(cls, cell_dir: str | Path) -> Segmenter:
        """Load from an artifact dir (``tokenizer.json`` + ``meta.yaml``)."""
        cell_dir = Path(cell_dir)
        spec = json.loads((cell_dir / "tokenizer.json").read_text())
        merge_brackets = _read_merge_brackets(cell_dir)
        return cls(spec, merge_brackets=merge_brackets)

    # -- public --------------------------------------------------------------
    def segment(self, smiles: str) -> list[Piece]:
        """Return the ordered output tokens for ``smiles``."""
        s = self._normalize(smiles)
        pieces: list[Piece] = []
        for text, start in self._pretokenize(s):
            if text.startswith("[") and text.endswith("]"):
                pieces.extend(self._bracket_pieces(text, start))
            else:
                # Organic run, `%dd`, a lone delimiter, or a bare digit: glyph-
                # split the whole pretoken and merge within it. Single-glyph
                # pretokens (`(`, `.`, a digit) merge to themselves, unchanged.
                pieces.extend(self._merge_run(self._organic_glyphs(text, start)))
        return pieces

    def _pretokenize(self, s: str) -> list[tuple[str, int]]:
        """Split ``s`` into ordered (text, start) pretokens, delimiters isolated.

        Reproduces the ``Split`` pre-tokenizer with ``behavior="Isolated"``:
        each regex match is its own pretoken and the gaps between matches are
        pretokens too, all kept in source order.
        """
        out: list[tuple[str, int]] = []
        pos = 0
        for m in self._pre.finditer(s):
            if m.start() > pos:
                out.append((s[pos : m.start()], pos))
            out.append((m.group(0), m.start()))
            pos = m.end()
        if pos < len(s):
            out.append((s[pos:], pos))
        return out

    def _organic_glyphs(self, text: str, start: int) -> list[_Glyph]:
        return [
            _Glyph(m.group(0), start + m.start(), start + m.end())
            for m in self._outer.finditer(text)
        ]

    def tokens(self, smiles: str) -> list[str]:
        return [p.text for p in self.segment(smiles)]

    def normalize(self, smiles: str) -> str:
        """The normalized SMILES that ``Piece`` offsets index into.

        Public so callers (e.g. the atom-overlay) can align piece char spans
        with the same string the segmenter saw.
        """
        return self._normalize(smiles)

    def glyphs(self, smiles: str) -> list[Piece]:
        """The atomic OpenSMILES glyph stream (the no-merge base decomposition).

        Same for both arms: merging never changes the glyph base, only how it is
        grouped. Used to align the two segmentations position-by-position (the
        nesting view). Each returned ``Piece`` is one glyph (``n_glyphs == 1``).
        """
        s = self._normalize(smiles)
        out: list[Piece] = []
        for text, start in self._pretokenize(s):
            if text.startswith("[") and text.endswith("]"):
                gl = self._bracket_glyphs(text, start)
            else:
                gl = self._organic_glyphs(text, start)
            out.extend(Piece(g.text, g.start, g.end, 1) for g in gl)
        return out

    # -- bracket handling ----------------------------------------------------
    def _bracket_pieces(self, atom: str, start: int) -> list[Piece]:
        glyphs = self._bracket_glyphs(atom, start)
        if self.merge_brackets:
            # whole [...] is one mergeable run
            return self._merge_run(glyphs)
        # NMB: '[' and ']' are barriers; inner glyphs merge among themselves
        out: list[Piece] = [Piece("[", start, start + 1, 1)]
        out.extend(self._merge_run(glyphs[1:-1]))
        end = start + len(atom)
        out.append(Piece("]", end - 1, end, 1))
        return out

    def _bracket_glyphs(self, atom: str, start: int) -> list[_Glyph]:
        """Decompose ``[...]`` into glyphs with absolute char offsets."""
        body = atom[1:-1]
        body_start = start + 1
        glyphs = [_Glyph("[", start, start + 1)]
        im = self._inner.fullmatch(body)
        if im is None:  # defensive: shouldn't happen on conformant SMILES
            glyphs.extend(
                _Glyph(c, body_start + i, body_start + i + 1)
                for i, c in enumerate(body)
            )
        else:
            for gi in range(1, (im.lastindex or 0) + 1):
                g = im.group(gi)
                if not g:  # skip unmatched (None) and empty-string groups
                    continue
                gs, _ = im.span(gi)
                if g.isdigit():
                    # every digit is its own glyph in the OpenSMILES base:
                    # isotope/charge/H-count/map-num numbers split per char.
                    for k, ch in enumerate(g):
                        off = body_start + gs + k
                        glyphs.append(_Glyph(ch, off, off + 1))
                else:
                    glyphs.append(_Glyph(g, body_start + gs, body_start + gs + len(g)))
        end = start + len(atom)
        glyphs.append(_Glyph("]", end - 1, end))
        return glyphs

    # -- model over one mergeable run ---------------------------------------
    def _merge_run(self, glyphs: list[_Glyph]) -> list[Piece]:
        if not glyphs:
            return []
        if self.algorithm == "bpe":
            return self._bpe(glyphs)
        return self._viterbi(glyphs)

    def _bpe(self, glyphs: list[_Glyph]) -> list[Piece]:
        """Merge one run using the HF ``tokenizers`` ``merge_word`` algorithm.

        A doubly-linked list of symbols plus a min-heap keyed by ``(rank, pos)``:
        pop the lowest-rank pair (leftmost on ties), merge in place, push the two
        new neighbouring pairs. Stale heap entries are skipped by re-checking
        adjacency and rank. This reproduces smirk's ordering exactly on runs of
        identical pairs, where a naive greedy rescan diverges.
        """
        n = len(glyphs)
        if n == 1:
            return [self._piece(glyphs[0])]
        text = [g.text for g in glyphs]
        start = [g.start for g in glyphs]
        end = [g.end for g in glyphs]
        prev = list(range(-1, n - 1))
        nxt = [*range(1, n), -1]
        alive = [True] * n

        heap: list[tuple[int, int, int]] = []
        for i in range(n - 1):
            r = self._rank.get((text[i], text[i + 1]))
            if r is not None:
                heap.append((r, i, i + 1))
        heapq.heapify(heap)

        while heap:
            rank, i, j = heapq.heappop(heap)
            if not (alive[i] and alive[j]) or nxt[i] != j:
                continue  # stale entry
            if self._rank.get((text[i], text[j])) != rank:
                continue  # pair no longer this rank
            text[i] += text[j]
            end[i] = end[j]
            alive[j] = False
            after = nxt[j]
            nxt[i] = after
            if after != -1:
                prev[after] = i
            p = prev[i]
            if p != -1:
                r = self._rank.get((text[p], text[i]))
                if r is not None:
                    heapq.heappush(heap, (r, p, i))
            if after != -1:
                r = self._rank.get((text[i], text[after]))
                if r is not None:
                    heapq.heappush(heap, (r, i, after))

        out: list[Piece] = []
        i = 0
        while i != -1:
            out.append(Piece(text[i], start[i], end[i], _glyph_count(text[i])))
            i = nxt[i]
        return out

    def _viterbi(self, glyphs: list[_Glyph]) -> list[Piece]:
        """Max-score Viterbi over the glyph run.

        - ``start`` ranges only over ``[k - max_piece_len, k)`` (span bound); a
          piece longer than any trained piece is never considered.
        - A single OOV glyph falls through to ``unk_score`` / the unk token;
          multi-glyph OOV spans are disallowed.
        - Ties keep the earliest ``start`` (strict ``>``, ``start`` ascending).

        KNOWN EDGE (≈0.01% of held-out SMILES, see ``validate.py``): when two
        segmentations have *bit-identical* total score, the shipped smirk binary
        occasionally resolves the tie the opposite way (e.g. ``CCCCC|C`` vs our
        ``C|CCCCC`` on long homopolymer chains). The fork's own ``UnigramModel``
        DP source agrees with us on strict-``>``; the deployed binary's Unigram
        inference clearly uses a different lattice tie-break that no fixed
        operator/direction reproduces. These cases never occur in real drawn
        molecules; they are irrelevant to the visual demo and are surfaced (not
        hidden) by the validation gate. BPE segmentation is exact.
        """
        n = len(glyphs)
        span = max(self._max_piece_len, 1)
        neg = float("-inf")
        best = [neg] * (n + 1)
        back = [-1] * (n + 1)
        is_unk = [False] * (n + 1)
        best[0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(0, i - span), i):
                if best[j] == neg:
                    continue
                piece = "".join(g.text for g in glyphs[j:i])
                sc = self._scores.get(piece)
                unk = False
                if sc is None:
                    if i - j != 1:
                        continue
                    sc, unk = self._unk_score, True
                if best[j] + sc > best[i]:
                    best[i], back[i], is_unk[i] = best[j] + sc, j, unk
        if best[n] == neg:  # defensive fallback: emit atomic glyphs
            return [self._piece(g) for g in glyphs]
        spans, i = [], n
        while i > 0:
            j = back[i]
            spans.append((j, i, is_unk[i]))
            i = j
        spans.reverse()
        return [
            Piece(
                self._unk_token if unk else "".join(g.text for g in glyphs[j:i]),
                glyphs[j].start,
                glyphs[i - 1].end,
                i - j,
            )
            for j, i, unk in spans
        ]

    @staticmethod
    def _piece(g: _Glyph) -> Piece:
        # a merged glyph tracks span but not internal count; recover count from
        # start/end only when needed - here n_glyphs is derived by the caller.
        return Piece(g.text, g.start, g.end, _glyph_count(g.text))

    def _normalize(self, smiles: str) -> str:
        s = smiles
        for pat, rep in self._replacements:
            s = s.replace(pat, rep)
        return s.strip() if self._strip else s


def _glyph_count(text: str) -> int:
    """Rough glyph count for a BPE piece (# of atomic units it concatenates).

    Only used for the ``is_multiglyph`` flag / display, never for segmentation.
    A single organic-subset char is one glyph; multi-char pieces are >1.
    """
    return 1 if len(text) == 1 else 2  # any concatenation is "multi"


def _build_merge_ranks(
    merges: list, vocab: dict
) -> tuple[dict[tuple[str, str], int], set[str]]:
    """Return ``(rank map, merged-token set)`` from the ordered merge list.

    Rank maps each ``(left, right)`` merge to its priority (lower = first).
    ``tokenizer.json`` stores merges as ``[id1, id2]`` pairs (older forks used
    ``"a b"`` strings). Merged tokens are *not* in ``model.vocab``; their ids are
    assigned sequentially after the base vocab, so we rebuild the id->token map
    on the fly as we walk the list. The merged tokens are exactly the BPE arm's
    multi-glyph vocabulary.
    """
    id2tok = {v: k for k, v in vocab.items()}
    next_id = len(id2tok)
    ranks: dict[tuple[str, str], int] = {}
    merged: set[str] = set()
    for i, mg in enumerate(merges):
        if isinstance(mg, str):
            a, b = mg.split(" ")
        else:
            a, b = id2tok[mg[0]], id2tok[mg[1]]
        ranks[(a, b)] = i
        id2tok[next_id] = a + b
        merged.add(a + b)
        next_id += 1
    return ranks, merged


def _read_merge_brackets(cell_dir: Path) -> bool:
    meta = cell_dir / "meta.yaml"
    if meta.is_file():
        for line in meta.read_text().splitlines():
            if line.startswith("merge_brackets:"):
                return line.split(":", 1)[1].strip() == "true"
    return cell_dir.name.endswith("_mb")
