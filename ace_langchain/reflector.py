"""
Reflector component using LangChain v1.0 best practices.

This module implements the Reflector role using LangChain's
Runnable interface with with_structured_output() for better reliability,
automatic validation, and improved error handling.

Enhanced features:
- Async support for concurrent processing
- Batch processing for multiple reflections
- Iterative refinement for higher quality analysis
- Enhanced formatting with code and tool call details
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from .playbook import LangChainPlaybook
from .prompts import REFLECTOR_PROMPT
from .types import BulletTag, ReflectorOutput, _format_optional, GeneratorOutput


# ================================
# Pydantic Models for Structured Output
# ================================

class BulletTagSchema(BaseModel):
    """Schema for a single bullet tag."""
    id: str = Field(description="Bullet ID")
    tag: str = Field(description="Tag: helpful, harmful, or neutral")
    justification: Optional[str] = Field(None, description="Specific evidence for this tag")


class ReflectorResponse(BaseModel):
    """Structured output schema for Reflector role."""
    reasoning: str = Field(description="Systematic analysis with numbered points")
    error_identification: str = Field(description="Specific error or 'none' if correct")
    error_location: Optional[str] = Field(None, description="Exact step where error occurred or 'N/A'")
    root_cause_analysis: str = Field(description="Underlying reason for error or success factor")
    correct_approach: str = Field(description="Detailed correct method with example")
    key_insight: str = Field(description="Reusable learning for future problems")
    confidence_in_analysis: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence in the analysis (0.0 to 1.0)"
    )
    bullet_tags: List[BulletTagSchema] = Field(
        default_factory=list, 
        description="Tags for bullets with justifications"
    )


class LangChainReflector:
    """
    Reflector role using LangChain with_structured_output() (v1.0 best practice).

    Analyzes generator outputs to extract lessons and improve strategies.
    Uses Pydantic models for automatic validation.
    
    Enhanced features:
    - Async support for concurrent processing
    - Batch processing for multiple reflections
    - Iterative refinement for higher quality analysis
    - Enhanced formatting with code and tool call details
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt_template: Optional[ChatPromptTemplate] = None,
        *,
        max_retries: int = 3,
        use_structured_output: bool = True,
        max_refinement_rounds: int = 2,
    ) -> None:
        """
        Initialize Reflector.

        Args:
            model: LangChain chat model
            prompt_template: Custom prompt template (uses REFLECTOR_PROMPT by default)
            max_retries: Maximum attempts if parsing fails (backward compatibility)
            use_structured_output: Use with_structured_output() if True (recommended)
            max_refinement_rounds: Maximum refinement rounds for iterative improvement (default: 2)
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
        self.prompt_template = prompt_template or REFLECTOR_PROMPT
        self.max_retries = max_retries
        self.use_structured_output = use_structured_output
        self.max_refinement_rounds = max_refinement_rounds

        if use_structured_output:
            structured_model = model.with_structured_output(
                ReflectorResponse,
                method="json_schema",
                include_raw=False,
            )
            self.chain = self.prompt_template | structured_model
        else:
            from langchain_core.output_parsers import JsonOutputParser
            self.json_parser = JsonOutputParser()
            self.chain = self.prompt_template | self.model | self.json_parser

    async def reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: LangChainPlaybook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        enable_refinement: bool = False,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """
        Reflect on generator output (async).

        Args:
            question: Original question
            generator_output: Generator's output to analyze
            playbook: Current playbook
            ground_truth: Correct answer if available
            feedback: Environment feedback
            enable_refinement: Whether to use iterative refinement (default: False)
            **kwargs: Additional arguments passed to the chain

        Returns:
            ReflectorOutput with analysis and bullet tags
        """
        if enable_refinement and self.max_refinement_rounds > 1:
            return await self.reflect_with_refinement(
                question=question,
                generator_output=generator_output,
                playbook=playbook,
                ground_truth=ground_truth,
                feedback=feedback,
                **kwargs,
            )
        
        return await self._reflect_once(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )

    async def _reflect_once(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: LangChainPlaybook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Single reflection pass (internal method)."""
        playbook_excerpt = _make_playbook_excerpt(playbook, generator_output.bullet_ids)
        
        # Enhanced formatting with code and tool calls
        generator_info = _format_generator_output_for_reflection(generator_output)

        base_input = {
            "question": question,
            "reasoning": generator_info,  # Use enhanced formatting
            "prediction": generator_output.final_answer,
            "ground_truth": _format_optional(ground_truth),
            "feedback": _format_optional(feedback),
            "playbook_excerpt": playbook_excerpt or "(no bullets referenced)",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self.use_structured_output:
                    result_obj: ReflectorResponse = await self.chain.ainvoke(base_input, **kwargs)
                    result = result_obj.model_dump()
                else:
                    result = await self.chain.ainvoke(base_input, **kwargs)
                    if not isinstance(result, dict):
                        from .types import _safe_json_loads
                        result = _safe_json_loads(str(result))

                bullet_tags: List[BulletTag] = []
                tags_payload = result.get("bullet_tags", [])
                if isinstance(tags_payload, Sequence):
                    for item in tags_payload:
                        if isinstance(item, dict):
                            bullet_tags.append(
                                BulletTag(
                                    id=str(item.get("id", "")),
                                    tag=str(item.get("tag", "")).lower(),
                                    justification=item.get("justification"),
                                )
                            )

                return ReflectorOutput(
                    reasoning=str(result.get("reasoning", "")),
                    error_identification=str(result.get("error_identification", "")),
                    root_cause_analysis=str(result.get("root_cause_analysis", "")),
                    correct_approach=str(result.get("correct_approach", "")),
                    key_insight=str(result.get("key_insight", "")),
                    bullet_tags=bullet_tags,
                    raw=result,
                    error_location=result.get("error_location"),
                    confidence_in_analysis=float(result.get("confidence_in_analysis")) if result.get("confidence_in_analysis") is not None else None,
                )
            except (ValueError, KeyError, TypeError, AttributeError) as err:
                last_error = err
                logger.warning(f"Reflection attempt {attempt + 1} failed: {err}")
                if attempt + 1 >= self.max_retries:
                    break
                base_input["question"] = (
                    base_input["question"]
                    + "\n\nPlease ensure your response is valid JSON with all required fields."
                )

        raise RuntimeError("Reflector failed to produce a result.") from last_error

    async def reflect_with_refinement(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: LangChainPlaybook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        max_rounds: Optional[int] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """
        Reflect with iterative refinement for higher quality analysis.

        Args:
            question: Original question
            generator_output: Generator's output to analyze
            playbook: Current playbook
            ground_truth: Correct answer if available
            feedback: Environment feedback
            max_rounds: Maximum refinement rounds (uses self.max_refinement_rounds if None)
            **kwargs: Additional arguments passed to the chain

        Returns:
            ReflectorOutput with refined analysis
        """
        max_rounds = max_rounds or self.max_refinement_rounds
        
        # Initial reflection
        reflection = await self._reflect_once(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )
        
        # Refinement rounds
        for round_num in range(1, max_rounds):
            try:
                refined = await self._refine_reflection(
                    reflection=reflection,
                    question=question,
                    generator_output=generator_output,
                    playbook=playbook,
                    round_num=round_num,
                    **kwargs,
                )
                
                if self._is_reflection_improved(reflection, refined):
                    reflection = refined
                    logger.info(f"Reflection improved in round {round_num}")
                else:
                    logger.info(f"Reflection did not improve in round {round_num}, stopping")
                    break
            except Exception as e:
                logger.warning(f"Refinement round {round_num} failed: {e}, using previous reflection")
                break
        
        return reflection

    async def _refine_reflection(
        self,
        *,
        reflection: ReflectorOutput,
        question: str,
        generator_output: GeneratorOutput,
        playbook: LangChainPlaybook,
        round_num: int,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Refine an existing reflection."""
        playbook_excerpt = _make_playbook_excerpt(playbook, generator_output.bullet_ids)
        generator_info = _format_generator_output_for_reflection(generator_output)
        
        # Create refinement prompt input
        base_input = {
            "question": question,
            "reasoning": f"""ORIGINAL REFLECTION (Round {round_num - 1}):
{reflection.reasoning}

ERROR IDENTIFICATION: {reflection.error_identification or 'Not specified'}
ROOT CAUSE: {reflection.root_cause_analysis or 'Not specified'}
CORRECT APPROACH: {reflection.correct_approach or 'Not specified'}
KEY INSIGHT: {reflection.key_insight or 'Not specified'}

GENERATOR OUTPUT:
{generator_info}""",
            "prediction": generator_output.final_answer,
            "ground_truth": "(refining previous analysis)",
            "feedback": "Please refine the reflection to make it more insightful and actionable.",
            "playbook_excerpt": playbook_excerpt or "(no bullets referenced)",
        }
        
        try:
            if self.use_structured_output:
                result_obj: ReflectorResponse = await self.chain.ainvoke(base_input, **kwargs)
                result = result_obj.model_dump()
            else:
                result = await self.chain.ainvoke(base_input, **kwargs)
                if not isinstance(result, dict):
                    from .types import _safe_json_loads
                    result = _safe_json_loads(str(result))
            
            # Parse bullet tags
            bullet_tags: List[BulletTag] = []
            tags_payload = result.get("bullet_tags", [])
            if isinstance(tags_payload, Sequence):
                for item in tags_payload:
                    if isinstance(item, dict):
                        bullet_tags.append(
                            BulletTag(
                                id=str(item.get("id", "")),
                                tag=str(item.get("tag", "")).lower(),
                                justification=item.get("justification"),
                            )
                        )
            else:
                # Keep original tags if not provided
                bullet_tags = reflection.bullet_tags
            
            return ReflectorOutput(
                reasoning=str(result.get("reasoning", reflection.reasoning)),
                error_identification=str(result.get("error_identification", reflection.error_identification)),
                root_cause_analysis=str(result.get("root_cause_analysis", reflection.root_cause_analysis)),
                correct_approach=str(result.get("correct_approach", reflection.correct_approach)),
                key_insight=str(result.get("key_insight", reflection.key_insight)),
                bullet_tags=bullet_tags,
                raw={**reflection.raw, "refinement_round": round_num, "refined_response": result},
                error_location=result.get("error_location", reflection.error_location),
                confidence_in_analysis=float(result.get("confidence_in_analysis")) if result.get("confidence_in_analysis") is not None else reflection.confidence_in_analysis,
            )
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return reflection  # Return original on failure

    def _is_reflection_improved(
        self,
        original: ReflectorOutput,
        refined: ReflectorOutput,
    ) -> bool:
        """
        Check if refined reflection is better than original.
        
        Uses heuristics:
        - More complete fields
        - Longer reasoning (but not excessively long)
        - More bullet tags with justifications
        """
        # Check completeness
        original_completeness = sum([
            1 for field in [
                original.error_identification,
                original.root_cause_analysis,
                original.correct_approach,
                original.key_insight
            ]
            if field and len(field.strip()) > 10
        ])
        
        refined_completeness = sum([
            1 for field in [
                refined.error_identification,
                refined.root_cause_analysis,
                refined.correct_approach,
                refined.key_insight
            ]
            if field and len(field.strip()) > 10
        ])
        
        # Check reasoning length (improvement should be longer but not excessive)
        original_len = len(original.reasoning)
        refined_len = len(refined.reasoning)
        
        # Check bullet tags with justifications
        original_tags_with_justification = sum(
            1 for tag in original.bullet_tags if tag.justification
        )
        refined_tags_with_justification = sum(
            1 for tag in refined.bullet_tags if tag.justification
        )
        
        # Improvement criteria:
        # 1. More complete fields, OR
        # 2. Longer reasoning (at least 10% increase, but not more than 200% increase), AND
        # 3. More tags with justifications
        completeness_improved = refined_completeness > original_completeness
        reasoning_improved = (
            refined_len > original_len * 1.1 and
            refined_len < original_len * 3.0  # Prevent excessive length
        )
        tags_improved = refined_tags_with_justification >= original_tags_with_justification
        
        return completeness_improved or (reasoning_improved and tags_improved)

    async def batch_reflect(
        self,
        reflections: List[Tuple[str, GeneratorOutput]],
        playbook: LangChainPlaybook,
        ground_truths: Optional[List[str]] = None,
        feedbacks: Optional[List[str]] = None,
        enable_refinement: bool = False,
        **kwargs: Any,
    ) -> List[ReflectorOutput]:
        """
        Reflect on multiple generator outputs concurrently.

        Args:
            reflections: List of (question, generator_output) tuples
            playbook: Current playbook
            ground_truths: Optional list of ground truth answers
            feedbacks: Optional list of feedback strings
            enable_refinement: Whether to use iterative refinement
            **kwargs: Additional arguments passed to reflect()

        Returns:
            List of ReflectorOutput objects
        """
        ground_truths = ground_truths or [None] * len(reflections)
        feedbacks = feedbacks or [None] * len(reflections)
        
        tasks = [
            self.reflect(
                question=question,
                generator_output=output,
                playbook=playbook,
                ground_truth=gt,
                feedback=fb,
                enable_refinement=enable_refinement,
                **kwargs,
            )
            for (question, output), gt, fb in zip(reflections, ground_truths, feedbacks)
        ]
        
        return await asyncio.gather(*tasks)


def _make_playbook_excerpt(
    playbook: LangChainPlaybook, bullet_ids: Sequence[str]
) -> str:
    """Create excerpt of playbook bullets referenced by generator."""
    lines: List[str] = []
    seen = set()
    for bullet_id in bullet_ids:
        if bullet_id in seen:
            continue
        bullets = playbook.bullets()
        bullet = next((b for b in bullets if b.id == bullet_id), None)
        if bullet:
            seen.add(bullet_id)
            # Determine tag based on helpful/harmful counts
            if bullet.helpful > bullet.harmful:
                tag = "HELPFUL"
            elif bullet.harmful > bullet.helpful:
                tag = "HARMFUL"
            else:
                tag = "NEUTRAL"
            lines.append(f"[{bullet.id}] [{tag}] {bullet.content}")
    return "\n".join(lines) if lines else ""


def _format_generator_output_for_reflection(
    generator_output: GeneratorOutput,
) -> str:
    """
    Format generator output with full details for reflection.
    
    Includes reasoning, code, tool calls, and confidence scores.
    """
    info = f"Reasoning Steps:\n{generator_output.reasoning}\n\n"
    info += f"Final Answer: {generator_output.final_answer}\n\n"
    
    if generator_output.generated_code:
        info += f"Generated Code:\n```\n{generator_output.generated_code}\n```\n\n"
    
    if generator_output.tool_calls:
        info += "Tool Calls:\n"
        for i, call in enumerate(generator_output.tool_calls, 1):
            tool_name = call.get("tool", "unknown")
            tool_args = call.get("arguments", {})
            tool_result = call.get("result", "N/A")
            success = call.get("success", False)
            status = "✓" if success else "✗"
            info += f"{i}. {status} {tool_name}({tool_args})\n"
            info += f"   Result: {tool_result}\n"
        info += "\n"
    
    if generator_output.confidence_scores:
        info += "Confidence Scores:\n"
        for bullet_id, score in generator_output.confidence_scores.items():
            info += f"  - Bullet {bullet_id}: {score:.2f}\n"
        info += "\n"
    
    if generator_output.answer_confidence is not None:
        info += f"Overall Answer Confidence: {generator_output.answer_confidence:.2f}\n"
    
    return info.strip()

