"""Evaluation and metric helpers shared by the experiment runners.

Submodules are imported directly; this package intentionally re-exports nothing
so that ``import DS_VI.experiments.metrics.attribute_metrics`` always resolves to the
module rather than a same-named function.

* :mod:`DS_VI.experiments.metrics.detection_metrics` — box AP/F1 and RGB/SAR ground-truth
  pairing, plus the cross-modal reference inference.
* :mod:`DS_VI.experiments.metrics.evaluate` — match (6-class box) and time (identity +
  change) evaluation entry points.
* :mod:`DS_VI.experiments.metrics.attribute_metrics` — attribute correctness/coverage.
* :mod:`DS_VI.experiments.metrics.property_metrics` — instance/object property metrics
  computed from the ``truth_*/tool_*_property.xlsx`` workbooks: detection (G1),
  attributes (G2), end-to-end attributes (G4), identity (G5), change (G6).
"""
