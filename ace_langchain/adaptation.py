"""
Adaptation loops using LangGraph StateGraph for workflow orchestration.

This module implements offline and online ACE adaptation using LangGraph's
StateGraph to manage the Generator -> Reflector -> Curator workflow.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, TypedDict

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore
    START = None  # type: ignore
    END = None  # type: ignore
    MemorySaver = None  # type: ignore

from .playbook import LangChainPlaybook
from .generator import LangChainGenerator
from .reflector import LangChainReflector
from .curator import LangChainCurator
from .types import (
    CuratorOutput,
    GeneratorOutput,
    ReflectorOutput,
)


@dataclass
class Sample:
    """Single task instance presented to ACE."""

    question: str
    context: str = ""
    ground_truth: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EnvironmentResult:
    """Feedback returned by the task environment after executing the generator output."""

    feedback: str
    ground_truth: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)


class TaskEnvironment(ABC):
    """
    Abstract interface for evaluating generator outputs.

    Implement this class to define how your specific task evaluates
    the Generator's answers. The environment provides feedback that
    helps ACE learn what works and what doesn't.
    """

    @abstractmethod
    def evaluate(
        self, sample: Sample, generator_output: GeneratorOutput
    ) -> EnvironmentResult:
        """
        Evaluate the generator's output for a given sample.

        Args:
            sample: The input sample with question and context
            generator_output: The Generator's produced answer

        Returns:
            EnvironmentResult with feedback and optional ground truth
        """


@dataclass
class AdapterStepResult:
    """Result from processing a single sample."""

    sample: Sample
    generator_output: GeneratorOutput
    environment_result: EnvironmentResult
    reflection: ReflectorOutput
    curator_output: CuratorOutput
    playbook_snapshot: str

    # Observability metadata
    epoch: int = 0
    step: int = 0
    performance_score: float = 0.0


# LangGraph State definition
class ACEState(TypedDict):
    """State managed by LangGraph for ACE workflow."""

    sample: Sample
    playbook: LangChainPlaybook
    generator_output: Optional[GeneratorOutput]
    environment_result: Optional[EnvironmentResult]
    reflection: Optional[ReflectorOutput]
    curator_output: Optional[CuratorOutput]
    recent_reflections: List[str]
    epoch: int
    step: int
    reflection_window: int


class LangChainOfflineAdapter:
    """
    Offline adapter using LangGraph StateGraph.

    Processes multiple samples in batches/epochs with the full
    Generator -> Reflector -> Curator workflow.
    """

    def __init__(
        self,
        *,
        playbook: Optional[LangChainPlaybook] = None,
        generator: LangChainGenerator,
        reflector: LangChainReflector,
        curator: LangChainCurator,
        max_refinement_rounds: int = 1,
        reflection_window: int = 3,
    ) -> None:
        """
        Initialize offline adapter.

        Args:
            playbook: Playbook instance (creates new if None)
            generator: Generator role instance
            reflector: Reflector role instance
            curator: Curator role instance
            max_refinement_rounds: Max reflection refinement rounds
            reflection_window: Number of recent reflections to keep
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required. Install with: pip install langgraph"
            )

        self.playbook = playbook or LangChainPlaybook()
        self.generator = generator
        self.reflector = reflector
        self.curator = curator
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window

        # Build LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        async def generator_node(state: ACEState) -> Dict:
            """Generate answer using playbook."""
            sample = state["sample"]
            reflection_context = "\n---\n".join(state["recent_reflections"])

            generator_output = await self.generator.generate(
                question=sample.question,
                context=sample.context,
                playbook=state["playbook"],
                reflection=reflection_context if reflection_context else None,
            )

            return {
                "generator_output": generator_output,
            }

        async def reflector_node(state: ACEState) -> Dict:
            """Reflect on generator output."""
            sample = state["sample"]
            generator_output = state["generator_output"]
            env_result = state["environment_result"]

            reflection = await self.reflector.reflect(
                question=sample.question,
                generator_output=generator_output,
                playbook=state["playbook"],
                ground_truth=env_result.ground_truth if env_result else None,
                feedback=env_result.feedback if env_result else None,
            )

            # Apply bullet tags
            for tag in reflection.bullet_tags:
                try:
                    state["playbook"].tag_bullet(tag.id, tag.tag)
                except ValueError:
                    continue

            # Update recent reflections
            recent_reflections = state["recent_reflections"].copy()
            recent_reflections.append(json.dumps(reflection.raw, ensure_ascii=False))
            if len(recent_reflections) > state["reflection_window"]:
                recent_reflections = recent_reflections[-state["reflection_window"] :]

            return {
                "reflection": reflection,
                "recent_reflections": recent_reflections,
            }

        async def curator_node(state: ACEState) -> Dict:
            """Curate playbook updates."""
            sample = state["sample"]
            env_result = state["environment_result"]
            reflection = state["reflection"]

            question_context = self._question_context(sample, env_result)
            progress = self._progress_string(
                state["epoch"], state["epoch"], state["step"], state["step"]
            )

            curator_output = await self.curator.curate(
                reflection=reflection,
                playbook=state["playbook"],
                question_context=question_context,
                progress=progress,
            )

            # Apply delta to playbook
            state["playbook"].apply_delta(curator_output.delta)

            return {
                "curator_output": curator_output,
            }

        # Build graph
        builder = StateGraph(ACEState)
        builder.add_node("generator", generator_node)
        builder.add_node("reflector", reflector_node)
        builder.add_node("curator", curator_node)

        # Define edges
        builder.add_edge(START, "generator")
        builder.add_edge("generator", "reflector")
        builder.add_edge("reflector", "curator")
        builder.add_edge("curator", END)

        # Compile with checkpointer for persistence
        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    def _question_context(
        self, sample: Sample, environment_result: Optional[EnvironmentResult]
    ) -> str:
        """Build question context string."""
        parts = [
            f"question: {sample.question}",
            f"context: {sample.context}",
            f"metadata: {json.dumps(sample.metadata)}",
        ]
        if environment_result:
            parts.extend(
                [
                    f"feedback: {environment_result.feedback}",
                    f"ground_truth: {environment_result.ground_truth}",
                ]
            )
        return "\n".join(parts)

    def _progress_string(
        self, epoch: int, total_epochs: int, step: int, total_steps: int
    ) -> str:
        """Format progress string."""
        return f"epoch {epoch}/{total_epochs} · sample {step}/{total_steps}"

    async def run(
        self,
        samples: Sequence[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
    ) -> List[AdapterStepResult]:
        """
        Run offline adaptation.

        Args:
            samples: List of samples to process
            environment: Environment for evaluating outputs
            epochs: Number of training epochs

        Returns:
            List of AdapterStepResult for each sample
        """
        results: List[AdapterStepResult] = []

        for epoch in range(epochs):
            for step, sample in enumerate(samples, 1):
                # Prepare initial state
                initial_state: ACEState = {
                    "sample": sample,
                    "playbook": self.playbook,
                    "generator_output": None,
                    "environment_result": None,
                    "reflection": None,
                    "curator_output": None,
                    "recent_reflections": [],
                    "epoch": epoch + 1,
                    "step": step,
                    "reflection_window": self.reflection_window,
                }

                # Run graph up to generator
                config = {"configurable": {"thread_id": f"epoch_{epoch}_step_{step}"}}
                state_after_generator = await self.graph.ainvoke(
                    initial_state, config=config, interrupt_after=["generator"]
                )

                # Evaluate with environment
                generator_output = state_after_generator["generator_output"]
                env_result = environment.evaluate(sample, generator_output)

                # Continue graph with environment result
                state_with_env: ACEState = {
                    **state_after_generator,
                    "environment_result": env_result,
                }

                # Run rest of graph
                final_state = await self.graph.ainvoke(
                    state_with_env, config=config, interrupt_after=["curator"]
                )

                # Calculate performance score
                performance_score = 0.0
                if env_result.metrics:
                    score_metrics = [
                        float(v)
                        for k, v in env_result.metrics.items()
                        if k
                        in [
                            "correct",
                            "efficient",
                            "success",
                            "accuracy",
                            "score",
                            "syntax_valid",
                            "contains_required",
                        ]
                        and isinstance(v, (int, float, bool))
                    ]
                    if score_metrics:
                        performance_score = sum(score_metrics) / len(score_metrics)

                result = AdapterStepResult(
                    sample=sample,
                    generator_output=final_state["generator_output"],
                    environment_result=env_result,
                    reflection=final_state["reflection"],
                    curator_output=final_state["curator_output"],
                    playbook_snapshot=final_state["playbook"].as_prompt(),
                    epoch=epoch + 1,
                    step=step,
                    performance_score=performance_score,
                )
                results.append(result)

        return results


class LangChainOnlineAdapter:
    """
    Online adapter using LangGraph StateGraph.

    Processes samples sequentially, adapting in real-time during inference.
    Similar to offline adapter but designed for streaming/online scenarios.
    """

    def __init__(
        self,
        *,
        playbook: Optional[LangChainPlaybook] = None,
        generator: LangChainGenerator,
        reflector: LangChainReflector,
        curator: LangChainCurator,
        max_refinement_rounds: int = 1,
        reflection_window: int = 3,
    ) -> None:
        """
        Initialize online adapter.

        Args:
            playbook: Playbook instance (creates new if None)
            generator: Generator role instance
            reflector: Reflector role instance
            curator: Curator role instance
            max_refinement_rounds: Max reflection refinement rounds
            reflection_window: Number of recent reflections to keep
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required. Install with: pip install langgraph"
            )

        self.playbook = playbook or LangChainPlaybook()
        self.generator = generator
        self.reflector = reflector
        self.curator = curator
        self.max_refinement_rounds = max_refinement_rounds
        self.reflection_window = reflection_window

        # Build LangGraph workflow (same as offline)
        self.graph = self._build_graph()
        self._step_counter = 0

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow (same as offline adapter)."""
        # Reuse the same graph structure
        adapter = LangChainOfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            max_refinement_rounds=self.max_refinement_rounds,
            reflection_window=self.reflection_window,
        )
        return adapter.graph

    async def run(
        self,
        samples: Sequence[Sample],
        environment: TaskEnvironment,
    ) -> List[AdapterStepResult]:
        """
        Run online adaptation (processes samples sequentially).

        Args:
            samples: List of samples to process
            environment: Environment for evaluating outputs

        Returns:
            List of AdapterStepResult for each sample
        """
        results: List[AdapterStepResult] = []

        for sample in samples:
            self._step_counter += 1

            # Prepare initial state
            initial_state: ACEState = {
                "sample": sample,
                "playbook": self.playbook,
                "generator_output": None,
                "environment_result": None,
                "reflection": None,
                "curator_output": None,
                "recent_reflections": [],
                "epoch": 1,  # Online doesn't use epochs
                "step": self._step_counter,
                "reflection_window": self.reflection_window,
            }

            # Run graph
            config = {"configurable": {"thread_id": f"online_step_{self._step_counter}"}}
            state_after_generator = await self.graph.ainvoke(
                initial_state, config=config, interrupt_after=["generator"]
            )

            # Evaluate
            generator_output = state_after_generator["generator_output"]
            env_result = environment.evaluate(sample, generator_output)

            # Continue graph
            state_with_env: ACEState = {
                **state_after_generator,
                "environment_result": env_result,
            }

            final_state = await self.graph.ainvoke(
                state_with_env, config=config, interrupt_after=["curator"]
            )

            # Calculate performance score
            performance_score = 0.0
            if env_result.metrics:
                score_metrics = [
                    float(v)
                    for k, v in env_result.metrics.items()
                    if k
                    in [
                        "correct",
                        "efficient",
                        "success",
                        "accuracy",
                        "score",
                        "syntax_valid",
                        "contains_required",
                    ]
                    and isinstance(v, (int, float, bool))
                ]
                if score_metrics:
                    performance_score = sum(score_metrics) / len(score_metrics)

            result = AdapterStepResult(
                sample=sample,
                generator_output=final_state["generator_output"],
                environment_result=env_result,
                reflection=final_state["reflection"],
                curator_output=final_state["curator_output"],
                playbook_snapshot=final_state["playbook"].as_prompt(),
                epoch=1,
                step=self._step_counter,
                performance_score=performance_score,
            )
            results.append(result)

        return results

