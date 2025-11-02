"""
Generator component using LangChain v1.0 best practices.

This module implements the Generator role using LangChain's
Runnable interface with with_structured_output() for better reliability,
automatic validation, and improved error handling.

Enhanced with async support, tool integration, batch processing, and code execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import BaseTool
    from langchain_core.messages import ToolMessage, AIMessage
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
    PYDANTIC_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    PYDANTIC_AVAILABLE = False
    BaseChatModel = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    BaseTool = None  # type: ignore
    BaseModel = None  # type: ignore
    Field = None  # type: ignore

from .playbook import LangChainPlaybook, LangChainBullet
from .prompts import (
    GENERATOR_PROMPT,
    get_generator_prompt_with_date,
    get_generator_react_prompt_with_date,
)
from .types import GeneratorOutput, _format_optional

logger = logging.getLogger(__name__)


# ================================
# Pydantic Model for Structured Output
# ================================

class GeneratorResponse(BaseModel):
    """Structured output schema for Generator role."""
    reasoning: str = Field(description="Detailed step-by-step chain of thought with numbered steps")
    bullet_ids: List[str] = Field(default_factory=list, description="List of bullet IDs used")
    confidence_scores: Optional[Dict[str, float]] = Field(
        None, 
        description="Confidence scores for each bullet ID"
    )
    final_answer: str = Field(description="Complete, direct answer to the question")
    answer_confidence: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Overall confidence in the answer (0.0 to 1.0)"
    )
    generated_code: Optional[str] = Field(
        None,
        description="Optional code solution if applicable"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of tool calls made (each with tool, arguments, and result summary)"
    )


class LangChainGenerator:
    """
    Generator role using LangChain with_structured_output() (v1.0 best practice).

    Produces answers using the current playbook of strategies.
    Uses Pydantic models for automatic validation and better reliability.
    
    Enhanced features:
    - Async support for concurrent processing
    - LangChain Tools integration (ReAct capability)
    - Batch processing for multiple queries
    - Code execution support (with custom executor)
    - Smart playbook formatting and filtering
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt_template: Optional[ChatPromptTemplate] = None,
        *,
        max_retries: int = 3,
        use_structured_output: bool = True,
        tools: Optional[List[BaseTool]] = None,
        min_bullet_helpfulness: int = 0,
        max_retrieved_bullets: int = 10,
    ) -> None:
        """
        Initialize Generator.

        Args:
            model: LangChain chat model (e.g., ChatOpenAI, ChatAnthropic)
            prompt_template: Custom prompt template (uses GENERATOR_PROMPT by default)
            max_retries: Maximum attempts if parsing fails (backward compatibility)
            use_structured_output: Use with_structured_output() if True (recommended)
            tools: Optional list of LangChain tools for ReAct capability
            min_bullet_helpfulness: Minimum helpful count to include bullet (default: 0)
            max_retrieved_bullets: Maximum bullets to retrieve from playbook (default: 10)
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
        # Use v2 prompt with current date if no custom template provided
        if prompt_template is None:
            self.prompt_template = get_generator_prompt_with_date()
        else:
            self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.use_structured_output = use_structured_output
        self.tools = tools or []
        self.min_bullet_helpfulness = min_bullet_helpfulness
        self.max_retrieved_bullets = max_retrieved_bullets

        # Create chain with structured output (LangChain v1.0 best practice)
        if use_structured_output:
            # Use with_structured_output for better reliability and validation
            structured_model = model.with_structured_output(
                GeneratorResponse,
                method="json_schema",  # Use native structured output when available
                include_raw=False,
            )
            self.chain = self.prompt_template | structured_model
        else:
            # Fallback to JsonOutputParser for backward compatibility
            from langchain_core.output_parsers import JsonOutputParser
            from .types import _safe_json_loads
            self.json_parser = JsonOutputParser()
            self.chain = self.prompt_template | self.model | self.json_parser
        
        # Create model with tools if tools are provided
        if self.tools:
            self.model_with_tools = self.model.bind_tools(self.tools)
        else:
            self.model_with_tools = None


    def _format_playbook_for_generation(self, bullets: List[LangChainBullet]) -> str:
        """
        Format bullets for inclusion in the generation prompt.
        
        Enhanced formatting with section grouping and tag display.
        """
        if not bullets:
            return "No relevant strategies found in playbook."
        
        content = "ACE Playbook:\n"
        content += "=" * 50 + "\n"
        
        # Group bullets by section
        sections: Dict[str, List[LangChainBullet]] = {}
        for bullet in bullets:
            if bullet.section not in sections:
                sections[bullet.section] = []
            sections[bullet.section].append(bullet)
        
        for section, section_bullets in sections.items():
            content += f"\n{section.upper()}:\n"
            content += "-" * 20 + "\n"
            for bullet in section_bullets:
                # Determine tag based on helpful/harmful counts
                if bullet.helpful > bullet.harmful:
                    tag = "HELPFUL"
                elif bullet.harmful > bullet.helpful:
                    tag = "HARMFUL"
                else:
                    tag = "NEUTRAL"
                content += f"[{bullet.id}] [{tag}] {bullet.content}\n"
        
        content += "=" * 50 + "\n"
        return content

    def _get_relevant_bullets(
        self,
        playbook: LangChainPlaybook,
        question: str,
    ) -> List[LangChainBullet]:
        """
        Retrieve and filter relevant bullets from playbook.
        
        Returns bullets that:
        1. Match keywords in the question
        2. Meet minimum helpfulness threshold
        3. Are limited to max_retrieved_bullets
        """
        all_bullets = playbook.bullets()
        
        # Simple keyword matching (can be enhanced with embeddings)
        question_lower = question.lower()
        question_words = [w for w in question_lower.split() if len(w) > 2]
        
        relevant_bullets = []
        for bullet in all_bullets:
            # Check if any question word appears in bullet content
            content_lower = bullet.content.lower()
            if any(word in content_lower for word in question_words):
                relevant_bullets.append(bullet)
        
        # Filter by helpfulness
        filtered_bullets = [
            bullet for bullet in relevant_bullets
            if bullet.helpful >= self.min_bullet_helpfulness
        ]
        
        # Sort by helpfulness (helpful - harmful) and limit
        filtered_bullets.sort(
            key=lambda b: (b.helpful - b.harmful, b.helpful),
            reverse=True
        )
        
        return filtered_bullets[:self.max_retrieved_bullets]

    async def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: LangChainPlaybook,
        reflection: Optional[str] = None,
        enable_tools: bool = True,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """
        Generate an answer using the playbook strategies (async).

        Args:
            question: The question to answer
            context: Additional context or requirements
            playbook: The current playbook of strategies
            reflection: Optional reflection from previous attempts
            enable_tools: Whether to use tools if available
            **kwargs: Additional arguments passed to the chain

        Returns:
            GeneratorOutput with reasoning, final_answer, and bullet_ids used
        """
        # Get relevant bullets with filtering
        relevant_bullets = self._get_relevant_bullets(playbook, question)
        playbook_content = self._format_playbook_for_generation(relevant_bullets)
        
        base_input = {
            "question": question,
            "context": _format_optional(context),
            "playbook": playbook_content,
            "reflection": _format_optional(reflection),
        }

        # If tools are enabled and available, use tool-enabled model
        if enable_tools and self.tools and self.model_with_tools:
            return await self._generate_with_tools_async(
                base_input, relevant_bullets, **kwargs
            )

        # Otherwise use standard structured output
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self.use_structured_output:
                    result_obj: GeneratorResponse = await self.chain.ainvoke(base_input, **kwargs)
                    result = result_obj.model_dump()
                else:
                    result = await self.chain.ainvoke(base_input, **kwargs)
                    if not isinstance(result, dict):
                        from .types import _safe_json_loads
                        result = _safe_json_loads(str(result))

                reasoning = str(result.get("reasoning", ""))
                final_answer = str(result.get("final_answer", ""))
                bullet_ids = [
                    str(item)
                    for item in result.get("bullet_ids", [])
                    if isinstance(item, (str, int))
                ]
                confidence_scores = result.get("confidence_scores")
                answer_confidence = result.get("answer_confidence")

                return GeneratorOutput(
                    reasoning=reasoning,
                    final_answer=final_answer,
                    bullet_ids=bullet_ids,
                    raw=result,
                    confidence_scores=confidence_scores if isinstance(confidence_scores, dict) else None,
                    answer_confidence=float(answer_confidence) if answer_confidence is not None else None,
                )
            except (ValueError, KeyError, TypeError, AttributeError) as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                base_input["question"] = (
                    base_input["question"]
                    + "\n\nPlease ensure your response is valid JSON with all required fields."
                )

        raise RuntimeError("Generator failed to produce valid output.") from last_error

    async def _generate_with_tools_async(
        self,
        base_input: Dict[str, Any],
        relevant_bullets: List[LangChainBullet],
        max_iterations: int = 5,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """
        Generate with tool support using ReAct pattern.
        
        Implements a simple ReAct loop:
        1. Model generates response (potentially with tool calls)
        2. Execute tool calls
        3. Pass results back to model
        4. Repeat until final answer or max iterations
        """
        if not self.model_with_tools:
            raise ValueError("Tools not available. Initialize with tools parameter.")

        from langchain_core.messages import HumanMessage
        from datetime import datetime

        # Use unified ReAct prompt from prompts.py
        react_prompt = get_generator_react_prompt_with_date()
        
        # Format prompt template - it returns a list of messages
        formatted_messages = react_prompt.format_messages(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            playbook=base_input['playbook'],
            question=base_input['question'],
            context=base_input['context'],
            reflection=base_input.get('reflection', '(none)'),
        )
        
        # Use formatted messages directly (includes system message from template)
        # Add user message if not already in template
        messages = list(formatted_messages)
        
        # Ensure we have a user message
        if len(messages) == 1:  # Only system message
            messages.append(HumanMessage(
                content=f"""Problem: {base_input['question']}

Context: {base_input['context']}

{base_input['reflection'] if base_input.get('reflection') != '(none)' else ''}

Please solve this problem using the playbook strategies, available tools, and your reasoning."""
            ))
        tool_calls_made = []

        for iteration in range(max_iterations):
            # Get model response
            ai_msg: AIMessage = await self.model_with_tools.ainvoke(messages, **kwargs)
            messages.append(ai_msg)

            # Check if model wants to call tools
            if not ai_msg.tool_calls:
                # No more tool calls, extract final answer
                final_text = ai_msg.content
                
                # Try to parse structured output
                try:
                    # Use structured output chain for final parsing
                    final_input = {
                        "question": base_input["question"],
                        "context": base_input["context"],
                        "playbook": base_input["playbook"],
                        "reflection": f"Tool execution results: {final_text}",
                    }
                    result_obj: GeneratorResponse = await self.chain.ainvoke(final_input, **kwargs)
                    result = result_obj.model_dump()
                except Exception:
                    # Fallback: construct result from messages
                    reasoning_parts = [msg.content for msg in messages if msg.content]
                    result = {
                        "reasoning": "\n".join(reasoning_parts),
                        "final_answer": final_text,
                        "bullet_ids": [b.id for b in relevant_bullets],
                        "tool_calls": tool_calls_made,
                    }

                return GeneratorOutput(
                    reasoning=str(result.get("reasoning", final_text)),
                    final_answer=str(result.get("final_answer", final_text)),
                    bullet_ids=[b.id for b in relevant_bullets],
                    raw=result,
                    confidence_scores=result.get("confidence_scores"),
                    answer_confidence=result.get("answer_confidence"),
                    tool_calls=tool_calls_made,
                )

            # Execute tool calls
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", "")

                # Find the tool
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool:
                    error_msg = f"Tool {tool_name} not found"
                    logger.warning(error_msg)
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))
                    tool_calls_made.append({
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": error_msg,
                        "success": False,
                    })
                    continue

                # Execute tool
                try:
                    tool_result = await tool.ainvoke(tool_args) if hasattr(tool, 'ainvoke') else tool.invoke(tool_args)
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))
                    tool_calls_made.append({
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": str(tool_result),
                        "success": True,
                    })
                    logger.info(f"Tool call succeeded: {tool_name}")
                except Exception as e:
                    error_msg = f"Tool execution error: {str(e)}"
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))
                    tool_calls_made.append({
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": error_msg,
                        "success": False,
                    })

        # Max iterations reached
        raise RuntimeError(f"Generator reached max iterations ({max_iterations}) without final answer.")

    async def batch_generate(
        self,
        questions: List[str],
        context: Optional[str],
        playbook: LangChainPlaybook,
        reflection: Optional[str] = None,
        enable_tools: bool = True,
        **kwargs: Any,
    ) -> List[GeneratorOutput]:
        """
        Generate answers for multiple questions concurrently.

        Args:
            questions: List of questions to answer
            context: Additional context (shared across all questions)
            playbook: The current playbook of strategies
            reflection: Optional reflection from previous attempts
            enable_tools: Whether to use tools if available
            **kwargs: Additional arguments passed to the chain

        Returns:
            List of GeneratorOutput objects
        """
        tasks = [
            self.generate(
                question=q,
                context=context,
                playbook=playbook,
                reflection=reflection,
                enable_tools=enable_tools,
                **kwargs,
            )
            for q in questions
        ]
        return await asyncio.gather(*tasks)

    async def execute_code(
        self,
        code: str,
        executor: Optional[Callable[[str], Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute generated code safely.

        Args:
            code: Python code to execute
            executor: Custom executor function (if None, code execution is disabled)
            
        Returns:
            Dict with execution result, success status, and error message if any
            
        Note:
            By default, code execution is disabled for security.
            Provide a custom executor function for safe code execution (e.g., sandboxed environment).
        """
        if not code or not code.strip():
            return {
                "success": True,
                "result": "No code to execute",
                "error": None,
            }

        if executor:
            try:
                result = await executor(code) if asyncio.iscoroutinefunction(executor) else executor(code)
                return {
                    "success": True,
                    "result": str(result),
                    "error": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "result": None,
                    "error": str(e),
                }
        else:
            # Code execution disabled by default for security
            logger.warning(
                "Code execution disabled. Provide a custom executor function for safe code execution."
            )
            return {
                "success": False,
                "result": None,
                "error": "Code execution disabled. Provide executor function.",
            }

