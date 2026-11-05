"""Stage-2 cascade confirmation: shared-identity VI over D-S matched candidates.

Two layers:

* :mod:`DS_VI.models.cascade.vi` — run shared-identity VI on dual-view detections
  (vi alone or restricted to Stage-1 D-S matched pairs).
* :mod:`DS_VI.models.cascade.attributes` — attach annotation-derived attribute claims
  from a rebuilt ctx dataset and turn the VI output back into prediction objects.

Public re-exports keep the historical ``DS_VI.models.cascade`` import surface stable.
"""
from DS_VI.models.cascade.attributes import (
    ATTRIBUTES,
    STATUS_ACCEPT,
    VIEW_TAG,
    CtxIndex,
    cascade_objects,
    observation_id_for,
    run_cascade_attr_pair,
)
from DS_VI.models.cascade.status import (
    MATCH_STATUSES,
    STATUSES_BY_FAMILY,
    STATUS_TO_CHANGE,
    TIME_STATUSES,
    object_status,
)
from DS_VI.models.cascade.vi import (
    BEST_CONFIG,
    build_observations,
    run_vi_pair,
    vi_boxes,
    vi_time_objects,
)

__all__ = [
    "ATTRIBUTES",
    "MATCH_STATUSES",
    "STATUSES_BY_FAMILY",
    "STATUS_TO_CHANGE",
    "TIME_STATUSES",
    "object_status",
    "STATUS_ACCEPT",
    "VIEW_TAG",
    "BEST_CONFIG",
    "CtxIndex",
    "build_observations",
    "cascade_objects",
    "observation_id_for",
    "run_cascade_attr_pair",
    "run_vi_pair",
    "vi_boxes",
    "vi_time_objects",
]
