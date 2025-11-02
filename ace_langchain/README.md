# ACE LangChain Implementation

Complete LangChain/LangGraph-based implementation of the Agentic Context Engine (ACE) framework.

## Overview

This implementation provides the same functionality as the original ACE framework (`ace/`) but built using:

- **LangChain v1.0** - For LLM interactions and chain composition
- **LangGraph** - For workflow orchestration and state management
- **LangGraph Store** - For persistent playbook storage

## Features

✅ **Fully Compatible API** - Drop-in replacement for original ACE  
✅ **LangChain Chains** - Uses `prompt | model | parser` pattern  
✅ **LangGraph Workflows** - StateGraph for Generator → Reflector → Curator flow  
✅ **Persistent Storage** - LangGraph Store for playbook persistence  
✅ **Modern Patterns** - Built on LangChain v1.0 best practices  

## Installation

```bash
# Core dependencies
pip install langchain langchain-core langgraph pydantic

# LLM provider integrations (choose one or more)
pip install langchain-openai        # OpenAI
pip install langchain-anthropic     # Anthropic Claude
pip install langchain-google-genai  # Google Gemini
```

**Note**: 
- This implementation uses **prompts_v2.py** (version 2.0) with enhanced structured outputs
- Uses **`with_structured_output()`** (LangChain v1.0 best practice) for better reliability
- Requires **pydantic** for automatic validation and type safety

## Quick Start

```python
from langchain_openai import ChatOpenAI
from ace_langchain import (
    LangChainPlaybook,
    LangChainGenerator,
    LangChainReflector,
    LangChainCurator,
    LangChainOfflineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
)

# Initialize LangChain model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Create ACE components (automatically uses with_structured_output())
playbook = LangChainPlaybook()
generator = LangChainGenerator(model)  # Uses Pydantic models for validation
reflector = LangChainReflector(model)  # Automatic type checking
curator = LangChainCurator(model)      # Better error handling

# Optional: Disable structured output for backward compatibility
# generator = LangChainGenerator(model, use_structured_output=False)

# Create adapter
adapter = LangChainOfflineAdapter(
    playbook=playbook,
    generator=generator,
    reflector=reflector,
    curator=curator,
)

# Define environment
class SimpleEnvironment(TaskEnvironment):
    def evaluate(self, sample, generator_output):
        correct = sample.ground_truth.lower() in generator_output.final_answer.lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )

# Prepare samples
samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="Capital of France?", ground_truth="Paris"),
]

# Run adaptation
environment = SimpleEnvironment()
results = adapter.run(samples, environment, epochs=1)

print(f"Learned {len(playbook.bullets())} strategies")
```

## Architecture

### Components

1. **LangChainPlaybook** - Uses LangGraph Store for persistence
2. **LangChainGenerator** - Uses LangChain Chains (`prompt | model | parser`)
3. **LangChainReflector** - Analyzes performance using LangChain
4. **LangChainCurator** - Updates playbook using LangChain
5. **LangChainOfflineAdapter** - Uses LangGraph StateGraph for workflow
6. **LangChainOnlineAdapter** - Sequential processing with LangGraph

### Workflow

```
┌─────────────┐
│   Sample    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│  Generator  │────▶│  Reflector   │
│  (LangChain)│     │  (LangChain) │
└─────────────┘     └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Curator    │
                    │  (LangChain) │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Playbook   │
                    │ (LangGraph   │
                    │    Store)    │
                    └──────────────┘
```

## Differences from Original ACE

| Feature | Original ACE | LangChain ACE |
|---------|-------------|---------------|
| LLM Client | Custom `LLMClient` interface | LangChain `BaseChatModel` |
| Prompts | String templates (v1 or v2) | `ChatPromptTemplate` (v2 only) |
| Chains | Custom implementation | LangChain `Runnable` chains |
| Workflow | Custom adapter loops | LangGraph `StateGraph` |
| Storage | JSON files | LangGraph `Store` |
| Memory | In-memory lists | LangGraph state management |
| Prompt Version | Supports v1 and v2 | **Only v2 prompts** (enhanced) |

## Key Features

### Prompt Version 2.0
This implementation uses **prompts_v2.py** exclusively, which includes:

- ✅ **Confidence Scores**: Generators output confidence levels for each strategy
- ✅ **Enhanced Error Handling**: Explicit error identification and location
- ✅ **Structured Analysis**: Better JSON schemas with validation
- ✅ **Evidence-Based Tagging**: Bullet tags include justification
- ✅ **Quality Control**: Explicit anti-patterns and quality thresholds

### LangChain v1.0 Best Practices
- ✅ **`with_structured_output()`**: Uses Pydantic models for automatic validation
- ✅ **Type Safety**: Full type checking with Pydantic BaseModel
- ✅ **Better Reliability**: Native structured output from model providers
- ✅ **Backward Compatible**: Can fallback to JsonOutputParser if needed

## Advantages

1. **Ecosystem Integration** - Works seamlessly with LangChain tools, agents, etc.
2. **Persistent State** - LangGraph checkpointer for resumable workflows
3. **Streaming Support** - Built-in streaming via LangChain
4. **Observability** - LangSmith integration for tracing
5. **Scalability** - LangGraph Server for production deployment

## Migration Guide

To migrate from original ACE to LangChain ACE:

```python
# Before (original ACE)
from ace import Generator, LiteLLMClient, Playbook
client = LiteLLMClient(model="gpt-4")
generator = Generator(client)

# After (LangChain ACE)
from langchain_openai import ChatOpenAI
from ace_langchain import LangChainGenerator, LangChainPlaybook
model = ChatOpenAI(model="gpt-4")
generator = LangChainGenerator(model)
```

## Examples

See `examples/` directory for complete examples.

## Features Comparison with Original ACE

| Feature | Original ACE | LangChain ACE |
|---------|-------------|---------------|
| **LLM Client** | Custom `LLMClient` interface | LangChain `BaseChatModel` |
| **Prompts** | String templates (v1 or v2) | `ChatPromptTemplate` (v2 only) |
| **Chains** | Custom implementation | LangChain `Runnable` chains |
| **Workflow** | Custom adapter loops | LangGraph `StateGraph` |
| **Storage** | JSON files | LangGraph `Store` |
| **Structured Output** | Manual JSON parsing | `with_structured_output()` with Pydantic |
| **ReAct Support** | ❌ No | ✅ Yes (via LangChain Tools) |
| **Async Support** | ⚠️ Partial | ✅ Full async |

## ReAct Pattern Support

The `LangChainGenerator` supports ReAct (Reasoning + Acting) pattern when tools are provided:

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # ... implementation
    return results

model = ChatOpenAI(model="gpt-4o-mini")
generator = LangChainGenerator(
    model=model,
    tools=[search_web],  # Enable ReAct
)

# Generator will automatically use tools when needed
output = await generator.generate(
    question="What is the current price of Bitcoin?",
    context="",
    playbook=playbook,
    enable_tools=True,  # Default: True
)

# Check tool usage
if output.tool_calls:
    for call in output.tool_calls:
        print(f"Tool: {call['tool']}, Result: {call['result']}")
```

**ReAct Flow:**
1. Model reasons about the problem
2. Decides to use tools if needed
3. Executes tools and observes results
4. Continues reasoning based on tool results
5. Returns final answer

See `REACT_IMPLEMENTATION.md` for detailed explanation.

## Status

✅ **All core features implemented**
- ✅ Playbook management with LangGraph Store
- ✅ Generator, Reflector, Curator roles with structured output
- ✅ Offline and Online adapters using LangGraph StateGraph
- ✅ v2 prompts with confidence scoring
- ✅ ReAct pattern support via LangChain Tools
- ✅ Full async support
- ✅ Type safety with Pydantic models

## License

Same as original ACE framework (MIT).

