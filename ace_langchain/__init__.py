"""
ACE Framework implementation using LangChain and LangGraph.

This module provides a complete LangChain/LangGraph-based implementation
of the Agentic Context Engine (ACE) framework, compatible with the original
ACE API but built on modern LangChain v1.0 patterns.
"""

from .playbook import LangChainPlaybook, LangChainBullet
from .generator import LangChainGenerator
from .reflector import LangChainReflector
from .curator import LangChainCurator
from .types import (
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
    BulletTag,
)
from .adaptation import (
    LangChainOfflineAdapter,
    LangChainOnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    AdapterStepResult,
)
from .delta import DeltaOperation, DeltaBatch
from .prompts import (
    GENERATOR_PROMPT,
    GENERATOR_REACT_PROMPT,
    REFLECTOR_PROMPT,
    CURATOR_PROMPT,
    get_generator_prompt_with_date,
    get_generator_react_prompt_with_date,
    PromptManager,
    validate_prompt_output,
)

# MCP integration (optional, requires langchain-mcp-adapters)
try:
    from .mcp_adapter import get_mcp_tools

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    get_mcp_tools = None  # type: ignore

__all__ = [
    # Playbook
    "LangChainPlaybook",
    "LangChainBullet",
    # Roles
    "LangChainGenerator",
    "LangChainReflector",
    "LangChainCurator",
    # Types
    "GeneratorOutput",
    "ReflectorOutput",
    "CuratorOutput",
    "BulletTag",
    # Adaptation
    "LangChainOfflineAdapter",
    "LangChainOnlineAdapter",
    "Sample",
    "TaskEnvironment",
    "EnvironmentResult",
    "AdapterStepResult",
    # Delta
    "DeltaOperation",
    "DeltaBatch",
    # Prompts
    "GENERATOR_PROMPT",
    "GENERATOR_REACT_PROMPT",
    "REFLECTOR_PROMPT",
    "CURATOR_PROMPT",
    "get_generator_prompt_with_date",
    "get_generator_react_prompt_with_date",
    "PromptManager",
    "validate_prompt_output",
    # MCP (optional, requires langchain-mcp-adapters)
    "get_mcp_tools",
    "MCP_AVAILABLE",
]

__version__ = "0.1.0"

