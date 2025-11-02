"""
Shared types and helper functions for ACE LangChain roles.

This module contains common data classes and utility functions
used across Generator, Reflector, and Curator components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BulletTag:
    """Tag for classifying bullet helpfulness (v2 compatible)."""
    id: str
    tag: str
    justification: Optional[str] = None


@dataclass
class GeneratorOutput:
    """Output from the Generator role (v2 compatible)."""
    reasoning: str
    final_answer: str
    bullet_ids: list[str]
    raw: Dict[str, Any]
    confidence_scores: Optional[Dict[str, float]] = None
    answer_confidence: Optional[float] = None
    generated_code: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ReflectorOutput:
    """Output from the Reflector role (v2 compatible)."""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: list[BulletTag]
    raw: Dict[str, Any]
    error_location: Optional[str] = None
    confidence_in_analysis: Optional[float] = None


@dataclass
class CuratorOutput:
    """Output from the Curator role."""
    delta: Any  # DeltaBatch - forward reference to avoid circular import
    raw: Dict[str, Any]


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Safely parse JSON with error logging."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        debug_path = Path("logs/json_failures.log")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as fh:
            fh.write("----\n")
            fh.write(repr(text))
            fh.write("\n")
        raise ValueError(f"LLM response is not valid JSON: {exc}\n{text}") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return data


def _format_optional(value: Optional[str]) -> str:
    """Format optional string value."""
    return value or "(none)"

