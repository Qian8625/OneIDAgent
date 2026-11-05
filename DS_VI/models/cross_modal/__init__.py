"""RGB/SAR cross-modal reference interpreter (model-only, no online service).

Re-exports :class:`CrossModalInterpreter`, used by the detection metrics and the
match-family baselines to build matched RGB/SAR ground truth and predictions.
"""
from DS_VI.models.cross_modal.interpreter import CrossModalInterpreter

__all__ = ["CrossModalInterpreter"]
