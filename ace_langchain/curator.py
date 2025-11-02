"""
Curator component using LangChain v1.0 best practices.

This module implements the Curator role using LangChain's
Runnable interface with with_structured_output() for better reliability,
automatic validation, and improved error handling.

Enhanced features:
- Async support for concurrent processing
- Batch processing for multiple curations
- Deduplication to avoid duplicate content
- Size limits to manage playbook growth
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
    PYDANTIC_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    PYDANTIC_AVAILABLE = False
    BaseChatModel = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    BaseModel = None  # type: ignore
    Field = None  # type: ignore

from .delta import DeltaBatch, DeltaOperation
from .playbook import LangChainPlaybook, LangChainBullet
from .prompts import CURATOR_PROMPT
from .types import CuratorOutput, ReflectorOutput


# ================================
# Pydantic Models for Structured Output
# ================================

class CuratorOperationSchema(BaseModel):
    """Schema for a single curator operation."""
    type: str = Field(description="Operation type: ADD, UPDATE, TAG, or REMOVE")
    section: Optional[str] = Field(None, description="Category like 'algebra', 'geometry', 'problem_solving'")
    content: Optional[str] = Field(None, description="Specific, actionable strategy with example")
    bullet_id: Optional[str] = Field(None, description="Required for UPDATE/TAG/REMOVE operations")
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Metadata with helpful/harmful counts and confidence"
    )
    justification: Optional[str] = Field(None, description="Why this operation improves the playbook")


class CuratorResponse(BaseModel):
    """Structured output schema for Curator role."""
    reasoning: str = Field(description="Analysis of what updates are needed and why")
    operations: List[CuratorOperationSchema] = Field(
        default_factory=list, 
        description="List of operations to perform"
    )


class LangChainCurator:
    """
    Curator role using LangChain with_structured_output() (v1.0 best practice).

    Transforms reflections into actionable playbook updates.
    Uses Pydantic models for automatic validation.
    
    Enhanced features:
    - Async support for concurrent processing
    - Batch processing for multiple curations
    - Deduplication to avoid duplicate content
    - Size limits to manage playbook growth
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt_template: Optional[ChatPromptTemplate] = None,
        *,
        max_retries: int = 3,
        use_structured_output: bool = True,
        enable_deduplication: bool = True,
        similarity_threshold: float = 0.8,
        max_playbook_size: Optional[int] = None,
    ) -> None:
        """
        Initialize Curator.

        Args:
            model: LangChain chat model
            prompt_template: Custom prompt template (uses CURATOR_PROMPT by default)
            max_retries: Maximum attempts if parsing fails (backward compatibility)
            use_structured_output: Use with_structured_output() if True (recommended)
            enable_deduplication: Enable content deduplication (default: True)
            similarity_threshold: Similarity threshold for deduplication (0.0-1.0, default: 0.8)
            max_playbook_size: Maximum number of bullets in playbook (None = unlimited)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required. Install with: pip install langchain langchain-core"
            )
        if use_structured_output and not PYDANTIC_AVAILABLE:
            raise ImportError(
                "Pydantic is required for structured output. Install with: pip install pydantic"
            )

        self.model = model
        self.prompt_template = prompt_template or CURATOR_PROMPT
        self.max_retries = max_retries
        self.use_structured_output = use_structured_output
        self.enable_deduplication = enable_deduplication
        self.similarity_threshold = similarity_threshold
        self.max_playbook_size = max_playbook_size

        if use_structured_output:
            structured_model = model.with_structured_output(
                CuratorResponse,
                method="json_schema",
                include_raw=False,
            )
            self.chain = self.prompt_template | structured_model
        else:
            from langchain_core.output_parsers import JsonOutputParser
            self.json_parser = JsonOutputParser()
            self.chain = self.prompt_template | self.model | self.json_parser

    async def curate(
        self,
        *,
        reflection: ReflectorOutput,
        playbook: LangChainPlaybook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> CuratorOutput:
        """
        Generate delta operations to update the playbook (async).

        Args:
            reflection: The Reflector's analysis
            playbook: Current playbook to potentially update
            question_context: Description of the task domain
            progress: Current progress summary
            **kwargs: Additional arguments passed to the chain

        Returns:
            CuratorOutput containing the delta operations to apply
        """
        base_input = {
            "progress": progress,
            "stats": json.dumps(playbook.stats()),
            "reflection": json.dumps(reflection.raw, ensure_ascii=False, indent=2),
            "playbook": playbook.as_prompt() or "(empty playbook)",
            "question_context": question_context,
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self.use_structured_output:
                    result_obj: CuratorResponse = await self.chain.ainvoke(base_input, **kwargs)
                    result = result_obj.model_dump()
                else:
                    result = await self.chain.ainvoke(base_input, **kwargs)
                    if not isinstance(result, dict):
                        from .types import _safe_json_loads
                        result = _safe_json_loads(str(result))

                # Convert operations to DeltaBatch format
                operations = []
                for op_data in result.get("operations", []):
                    if isinstance(op_data, dict):
                        op_type = op_data.get("type", "").upper()
                        if op_type == "ADD":
                            operations.append(
                                DeltaOperation(
                                    type="ADD",
                                    section=op_data.get("section", ""),
                                    content=op_data.get("content"),
                                    metadata=op_data.get("metadata") or {},
                                )
                            )
                        elif op_type in ("UPDATE", "TAG"):
                            bullet_id = op_data.get("bullet_id", "")
                            metadata = op_data.get("metadata") or {}
                            # Ensure metadata values are integers
                            clean_metadata = {
                                k: int(v) if isinstance(v, (int, float)) else 0
                                for k, v in metadata.items()
                            }
                            operations.append(
                                DeltaOperation(
                                    type=op_type,
                                    section=op_data.get("section", ""),
                                    bullet_id=bullet_id,
                                    content=op_data.get("content"),
                                    metadata=clean_metadata,
                                )
                            )
                        elif op_type == "REMOVE":
                            bullet_id = op_data.get("bullet_id", "")
                            operations.append(
                                DeltaOperation(
                                    type="REMOVE",
                                    section=op_data.get("section", ""),
                                    bullet_id=bullet_id,
                                )
                            )

                # Apply deduplication if enabled
                if self.enable_deduplication:
                    operations = await self._filter_duplicates(operations, playbook)

                # Apply size limits if configured
                if self.max_playbook_size:
                    operations = await self._apply_size_limits(operations, playbook)

                delta = DeltaBatch(
                    reasoning=result.get("reasoning", ""),
                    operations=operations
                )
                return CuratorOutput(delta=delta, raw=result)
            except (ValueError, KeyError, TypeError, AttributeError) as err:
                last_error = err
                logger.warning(f"Curation attempt {attempt + 1} failed: {err}")
                if attempt + 1 >= self.max_retries:
                    break
                base_input["progress"] = (
                    base_input["progress"]
                    + "\n\nPlease ensure your response is valid JSON with all required fields."
                )

        raise RuntimeError("Curator failed to produce valid output.") from last_error

    async def _filter_duplicates(
        self,
        operations: List[DeltaOperation],
        playbook: LangChainPlaybook,
    ) -> List[DeltaOperation]:
        """Filter out operations that would create duplicate content."""
        if not operations:
            return operations

        existing_bullets = playbook.bullets()
        filtered_operations = []

        for op in operations:
            if op.type == "ADD" and op.content:
                if not self._is_duplicate_content(op.content, existing_bullets):
                    filtered_operations.append(op)
                else:
                    logger.info(f"Skipping duplicate content: {op.content[:50]}...")
            else:
                # Keep UPDATE, TAG, REMOVE operations
                filtered_operations.append(op)

        return filtered_operations

    def _is_duplicate_content(
        self,
        new_content: str,
        existing_bullets: List[LangChainBullet],
        similarity_threshold: Optional[float] = None,
    ) -> bool:
        """Check if content is too similar to existing bullets."""
        threshold = similarity_threshold or self.similarity_threshold
        new_words = set(new_content.lower().split())

        for bullet in existing_bullets:
            bullet_words = set(bullet.content.lower().split())

            if not bullet_words or not new_words:
                continue

            similarity = self._content_similarity(new_words, bullet_words)
            if similarity >= threshold:
                return True

        return False

    def _content_similarity(self, words1: set, words2: set) -> float:
        """Calculate Jaccard similarity between two word sets."""
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    async def _apply_size_limits(
        self,
        operations: List[DeltaOperation],
        playbook: LangChainPlaybook,
    ) -> List[DeltaOperation]:
        """Apply size limits by removing low-scoring bullets if needed."""
        if not self.max_playbook_size:
            return operations

        current_size = len(playbook.bullets())
        add_operations = [op for op in operations if op.type == "ADD"]
        other_operations = [op for op in operations if op.type != "ADD"]

        # Count how many ADD operations we have
        new_adds = len(add_operations)

        # If adding would exceed limit, remove lowest-scoring bullets first
        if current_size + new_adds > self.max_playbook_size:
            excess = current_size + new_adds - self.max_playbook_size
            remove_operations = await self._generate_remove_operations(
                playbook, excess
            )
            # Add REMOVE operations before ADD operations
            return remove_operations + add_operations + other_operations

        return operations

    async def _generate_remove_operations(
        self,
        playbook: LangChainPlaybook,
        count: int,
    ) -> List[DeltaOperation]:
        """Generate REMOVE operations for lowest-scoring bullets."""
        bullets = playbook.bullets()

        if not bullets:
            return []

        # Score bullets by helpfulness (helpful - harmful)
        scored_bullets = [
            (bullet.helpful - bullet.harmful, bullet) for bullet in bullets
        ]

        # Sort by score (ascending - lowest scores first)
        scored_bullets.sort(key=lambda x: x[0])

        # Generate REMOVE operations for lowest-scoring bullets
        remove_operations = []
        for score, bullet in scored_bullets[:count]:
            remove_operations.append(
                DeltaOperation(
                    type="REMOVE",
                    section=bullet.section,
                    bullet_id=bullet.id,
                )
            )
            logger.info(
                f"Planning to remove low-scoring bullet [{bullet.id}] "
                f"(score: {score}) to make room for new content"
            )

        return remove_operations

    async def batch_curate(
        self,
        reflections: List[ReflectorOutput],
        playbook: LangChainPlaybook,
        question_contexts: Optional[List[str]] = None,
        progress_strings: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[CuratorOutput]:
        """
        Curate multiple reflections concurrently.

        Args:
            reflections: List of ReflectorOutput objects to curate
            playbook: Current playbook to potentially update
            question_contexts: Optional list of question context strings
            progress_strings: Optional list of progress strings
            **kwargs: Additional arguments passed to curate()

        Returns:
            List of CuratorOutput objects
        """
        question_contexts = question_contexts or [""] * len(reflections)
        progress_strings = progress_strings or [""] * len(reflections)

        tasks = [
            self.curate(
                reflection=reflection,
                playbook=playbook,
                question_context=ctx,
                progress=prog,
                **kwargs,
            )
            for reflection, ctx, prog in zip(
                reflections, question_contexts, progress_strings
            )
        ]

        return await asyncio.gather(*tasks)

