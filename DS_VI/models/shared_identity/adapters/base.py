"""Adapter protocol and tool-name based dispatch."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

from DS_VI.models.shared_identity.schema import Claim, PrimitiveObservation, Relation


@dataclass
class AdapterResult:
    """What an adapter extracted from one tool response."""

    observations: List[PrimitiveObservation] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    coverage: Dict[str, Any] = field(default_factory=dict)
    skipped_reason: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.observations and not self.claims and not self.relations


class BaseAdapter:
    """Base class for tool-response adapters.

    Subclasses declare the tool names they handle and implement
    :meth:`to_observations`.  Failed or empty responses must return an empty
    :class:`AdapterResult` with a ``skipped_reason``, never raise.
    """

    tool_names: Tuple[str, ...] = ()
    modality: str = "unknown"

    @classmethod
    def handles(cls, tool_name: str) -> bool:
        needle = str(tool_name or "").strip().lower()
        return any(needle == name.lower() for name in cls.tool_names)

    @staticmethod
    def response_failed(response: Mapping[str, Any]) -> Optional[str]:
        """Return a reason string when ``response`` must not produce claims."""
        if not isinstance(response, Mapping):
            return "response is not a mapping"
        error_code = response.get("error_code")
        if error_code not in (None, 0, "0"):
            return f"error_code={error_code}"
        structured = response.get("structured")
        status = None
        if isinstance(structured, Mapping):
            status = structured.get("status")
        status = status or response.get("status")
        if status is not None and str(status).lower() in {
            "error",
            "invalid_request",
            "invalid_dependency",
            "missing_inputs",
            "example_leakage_detected",
        }:
            return f"status={status}"
        return None

    def to_observations(
        self,
        tool_name: str,
        response: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> AdapterResult:  # pragma: no cover - abstract
        raise NotImplementedError


_REGISTRY: List[Type[BaseAdapter]] = []


def register(adapter_cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
    """Register an adapter class for tool-name dispatch."""
    if adapter_cls not in _REGISTRY:
        _REGISTRY.append(adapter_cls)
    return adapter_cls


def resolve(tool_name: str) -> Optional[BaseAdapter]:
    """Return an adapter instance for ``tool_name``, or ``None``."""
    for adapter_cls in _REGISTRY:
        if adapter_cls.handles(tool_name):
            return adapter_cls()
    return None


def registered_adapters() -> Sequence[Type[BaseAdapter]]:
    return tuple(_REGISTRY)


__all__ = ["AdapterResult", "BaseAdapter", "register", "resolve", "registered_adapters"]
