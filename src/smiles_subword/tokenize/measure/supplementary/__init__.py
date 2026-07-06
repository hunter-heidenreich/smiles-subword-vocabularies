"""Supplementary analyses beyond the seven headline measurements.

The seven frozen measurements (Deadzone, Absorption, Scaffold, Fertility,
Jaccard, Distribution, Segmentation) live in the parent ``measure`` package. This
subpackage holds the secondary analyses that support but are not part of that
headline set:

- ``transfer/`` — cross-corpus transfer matrix: ``math``, ``runner``, and the
  ``ood`` out-of-distribution extension on adversarial chemistry.
- ``sensitivity/`` — robustness-extras sensitivity analysis
  (``math``/``runner``/``io``).
- ``extras_audits`` — robustness-extras knob-inertness + merge-exhaustion
  summaries, deposited under ``results/data/audits/``.
- ``vocab_characterization`` — descriptive vocabulary statistics.
- ``marginal_jaccard`` — cross-arm overlap of pieces added between V-steps.
"""
