"""Adapters translating tool-worker responses into observations and claims.

Each adapter exposes ``to_observations(tool_name, response, context)`` returning
``(observations, claims, relations)``.  The EvidenceStore only knows this
protocol, so it never couples to a worker's private field names.
"""
from DS_VI.models.shared_identity.adapters.base import AdapterResult, register, resolve
from DS_VI.models.shared_identity.adapters.cross_modal import CrossModalAdapter
from DS_VI.models.shared_identity.adapters.optical import OpticalAdapter
from DS_VI.models.shared_identity.adapters.sar import SARAdapter
from DS_VI.models.shared_identity.adapters.temporal import TemporalAdapter

__all__ = [
    "AdapterResult",
    "CrossModalAdapter",
    "OpticalAdapter",
    "SARAdapter",
    "TemporalAdapter",
    "register",
    "resolve",
]
