"""Shared result protocol for agent nodes.

Agent nodes still return plain dictionaries so they remain compatible with
LangGraph and the existing Streamlit tabs.  The helpers below add a consistent
``agent_result`` envelope next to the legacy fields, making failures, traces,
and metadata easier to inspect without forcing a large migration all at once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class AgentResult:
    ok: bool
    agent: str
    stage: str
    data: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentTimer:
    """Small timing helper for recording per-agent latency."""

    def __init__(self) -> None:
        self._start = perf_counter()

    def elapsed_ms(self) -> int:
        return int((perf_counter() - self._start) * 1000)


def success_result(
    *,
    agent: str,
    stage: str,
    data: dict[str, Any],
    trace: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    legacy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return legacy fields plus a normalized successful agent envelope."""
    result = AgentResult(
        ok=True,
        agent=agent,
        stage=stage,
        data=data,
        trace=trace or [],
        error=None,
        metadata=metadata or {},
    ).to_dict()
    return {**(legacy or {}), "agent_result": result}


def failure_result(
    *,
    agent: str,
    stage: str,
    error: str,
    data: dict[str, Any] | None = None,
    trace: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    legacy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return legacy fields plus a normalized failed agent envelope."""
    result = AgentResult(
        ok=False,
        agent=agent,
        stage=stage,
        data=data or {},
        trace=trace or [],
        error=error,
        metadata=metadata or {},
    ).to_dict()
    return {**(legacy or {}), "agent_result": result}


def model_metadata(config: dict, key: str = "model_name", **extra: Any) -> dict[str, Any]:
    """Build lightweight metadata without depending on provider internals."""
    metadata = {"model": config.get(key, "unknown")}
    metadata.update(extra)
    return metadata

