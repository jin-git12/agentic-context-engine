"""
MCP (Model Context Protocol) Integration for ACE LangChain.

This module provides simple integration with official MultiServerMCPClient,
enabling the Generator to use MCP tools in its ReAct workflow.

Uses official langchain-mcp-adapters library (recommended by LangChain).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    from langchain_core.tools import BaseTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = None  # type: ignore

# Official MultiServerMCPClient (required)
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MultiServerMCPClient = None  # type: ignore


# ================================
# Simplified MCP Integration
# ================================


async def get_mcp_tools(
    servers_config: Dict[str, Dict[str, Any]],
) -> List[BaseTool]:
    """
    Get MCP tools from server configuration using official MultiServerMCPClient.

    This is the recommended way to use MCP with ACE LangChain.

    Args:
        servers_config: Dictionary of server configurations
            Each server config can have:
            - "transport": "stdio" | "streamable_http" | "sse"
            - "command": Command to run (for stdio)
            - "args": Arguments for command (for stdio)
            - "url": Server URL (for http/sse)
            - "headers": Optional headers (for http/sse)
        
    Returns:
        List of LangChain BaseTool instances ready to use with Generator

    Example:
        ```python
        from ace_langchain import get_mcp_tools, LangChainGenerator, LangChainPlaybook
        from langchain_openai import ChatOpenAI

        # Get tools from MCP servers
        tools = await get_mcp_tools({
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": ["/path/to/math_server.py"],
            },
            "weather": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            }
        })

        # Use with Generator
        model = ChatOpenAI(model="gpt-4o-mini")
        generator = LangChainGenerator(model=model, tools=tools)
        
        playbook = LangChainPlaybook()
        result = await generator.generate(
            question="What's (3 + 5) x 12?",
            context="",
            playbook=playbook,
            enable_tools=True,
        )
        ```

    Raises:
        ImportError: If langchain-mcp-adapters is not installed
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "langchain-mcp-adapters is required. Install with: pip install langchain-mcp-adapters"
        )

    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required. Install with: pip install langchain langchain-core"
        )

    client = MultiServerMCPClient(servers_config)
    tools = await client.get_tools()
    logger.info(f"Loaded {len(tools)} MCP tools from {len(servers_config)} server(s)")
    return tools

