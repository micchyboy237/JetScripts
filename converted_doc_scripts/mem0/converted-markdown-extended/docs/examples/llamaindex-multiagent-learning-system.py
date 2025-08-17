import asyncio
from jet.transformers.formatters import format_json
from datetime import datetime
from dotenv import load_dotenv
from jet.llm.ollama.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.memory.mem0 import Mem0Memory
import asyncio
import os
import shutil
import warnings


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: LlamaIndex Multi-Agent Learning System
---

<Snippet file="blank-notif.mdx" />

Build an intelligent multi-agent learning system that uses Mem0 to maintain persistent memory across multiple specialized agents. This example demonstrates how to create a tutoring system where different agents collaborate while sharing a unified memory layer.

## Overview

This example showcases a **Multi-Agent Personal Learning System** that combines:
- **LlamaIndex AgentWorkflow** for multi-agent orchestration
- **Mem0** for persistent, shared memory across agents
- **Multi-agents** that collaborate on teaching tasks

The system consists of two agents:
- **TutorAgent**: Primary instructor for explanations and concept teaching
- **PracticeAgent**: Generates exercises and tracks learning progress

Both agents share the same memory context, enabling seamless collaboration and continuous learning from student interactions.

## Key Features

- **Persistent Memory**: Agents remember previous interactions across sessions
- **Multi-Agent Collaboration**: Agents can hand off tasks to each other
- **Personalized Learning**: Adapts to individual student needs and learning styles
- **Progress Tracking**: Monitors learning patterns and skill development
- **Memory-Driven Teaching**: References past struggles and successes

## Prerequisites

Install the required packages:
"""
logger.info("## Overview")

pip install llama-index-core llama-index-memory-mem0 openai python-dotenv

"""
Set up your environment variables:
- `MEM0_API_KEY`: Your Mem0 Platform API key
# - `OPENAI_API_KEY`: Your MLX API key

You can obtain your Mem0 Platform API key from the [Mem0 Platform](https://app.mem0.ai).

## Complete Implementation
"""
logger.info("## Complete Implementation")

"""
Multi-Agent Personal Learning System: Mem0 + LlamaIndex AgentWorkflow Example

INSTALLATIONS:
!pip install llama-index-core llama-index-memory-mem0 openai

# You need MEM0_API_KEY and OPENAI_API_KEY to run the example.
"""




warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()


class MultiAgentLearningSystem:
    """
    Multi-Agent Architecture:
    - TutorAgent: Main teaching and explanations
    - PracticeAgent: Exercises and skill reinforcement
    - Shared Memory: Both agents learn from student interactions
    """

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.llm = MLX(model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.2)

        self.memory_context = {"user_id": student_id, "app": "learning_assistant"}
        self.memory = Mem0Memory.from_client(
            context=self.memory_context
        )

        self._setup_agents()

    def _setup_agents(self):
        """Setup two agents that work together and share memory"""

        async def assess_understanding(topic: str, student_response: str) -> str:
            """Assess student's understanding of a topic and save insights"""
            if "confused" in student_response.lower() or "don't understand" in student_response.lower():
                assessment = f"STRUGGLING with {topic}: {student_response}"
                insight = f"Student needs more help with {topic}. Prefers step-by-step explanations."
            elif "makes sense" in student_response.lower() or "got it" in student_response.lower():
                assessment = f"UNDERSTANDS {topic}: {student_response}"
                insight = f"Student grasped {topic} quickly. Can move to advanced concepts."
            else:
                assessment = f"PARTIAL understanding of {topic}: {student_response}"
                insight = f"Student has basic understanding of {topic}. Needs reinforcement."

            return f"Assessment: {assessment}\nInsight saved: {insight}"

        async def track_progress(topic: str, success_rate: str) -> str:
            """Track learning progress and identify patterns"""
            progress_note = f"Progress on {topic}: {success_rate} - {datetime.now().strftime('%Y-%m-%d')}"
            return f"Progress tracked: {progress_note}"

        tools = [
            FunctionTool.from_defaults(async_fn=assess_understanding),
            FunctionTool.from_defaults(async_fn=track_progress)
        ]

        self.tutor_agent = FunctionAgent(
            name="TutorAgent",
            description="Primary instructor that explains concepts and adapts to student needs",
            system_prompt="""
            You are a patient, adaptive programming tutor. Your key strength is REMEMBERING and BUILDING on previous interactions.

            Key Behaviors:
            1. Always check what the student has learned before (use memory context)
            2. Adapt explanations based on their preferred learning style
            3. Reference previous struggles or successes
            4. Build progressively on past lessons
            5. Use assess_understanding to evaluate responses and save insights

            MEMORY-DRIVEN TEACHING:
            - "Last time you struggled with X, so let's approach Y differently..."
            - "Since you prefer visual examples, here's a diagram..."
            - "Building on the functions we covered yesterday..."

            When student shows understanding, hand off to PracticeAgent for exercises.
            """,
            tools=tools,
            llm=self.llm,
            can_handoff_to=["PracticeAgent"]
        )

        self.practice_agent = FunctionAgent(
            name="PracticeAgent",
            description="Creates practice exercises and tracks progress based on student's learning history",
            system_prompt="""
            You create personalized practice exercises based on the student's learning history and current level.

            Key Behaviors:
            1. Generate problems that match their skill level (from memory)
            2. Focus on areas they've struggled with previously
            3. Gradually increase difficulty based on their progress
            4. Use track_progress to record their performance
            5. Provide encouraging feedback that references their growth

            MEMORY-DRIVEN PRACTICE:
            - "Let's practice loops again since you wanted more examples..."
            - "Here's a harder version of the problem you solved yesterday..."
            - "You've improved a lot in functions, ready for the next level?"

            After practice, can hand back to TutorAgent for concept review if needed.
            """,
            tools=tools,
            llm=self.llm,
            can_handoff_to=["TutorAgent"]
        )

        self.workflow = AgentWorkflow(
            agents=[self.tutor_agent, self.practice_agent],
            root_agent=self.tutor_agent.name,
            initial_state={
                "current_topic": "",
                "student_level": "beginner",
                "learning_style": "unknown",
                "session_goals": []
            }
        )

    async def start_learning_session(self, topic: str, student_message: str = "") -> str:
        """
        Start a learning session with multi-agent memory-aware teaching
        """

        if student_message:
            request = f"I want to learn about {topic}. {student_message}"
        else:
            request = f"I want to learn about {topic}."

        async def async_func_142():
            response = await self.workflow.run(
                user_msg=request,
                memory=self.memory
            )
            return response
        response = asyncio.run(async_func_142())
        logger.success(format_json(response))

        return str(response)

    async def get_learning_history(self) -> str:
        """Show what the system remembers about this student"""
        try:
            memories = self.memory.search(
                user_id=self.student_id,
                query="learning machine learning"
            )

            if memories and memories.get('results'):
                history = "\n".join(f"- {m['memory']}" for m in memories['results'])
                return history
            else:
                return "No learning history found yet. Let's start building your profile!"

        except Exception as e:
            return f"Memory retrieval error: {str(e)}"


async def run_learning_agent():

    learning_system = MultiAgentLearningSystem(student_id="Alexander")

    logger.debug("Session 1:")
    async def async_func_172():
        response = await learning_system.start_learning_session(
            "Vision Language Models",
            "I'm new to machine learning but I have good hold on Python and have 4 years of work experience.")
        return response
    response = asyncio.run(async_func_172())
    logger.success(format_json(response))
    logger.debug(response)

    logger.debug("\nSession 2:")
    async def async_func_178():
        response2 = await learning_system.start_learning_session(
            "Machine Learning", "what all did I cover so far?")
        return response2
    response2 = asyncio.run(async_func_178())
    logger.success(format_json(response2))
    logger.debug(response2)

    logger.debug("\nLearning History:")
    async def run_async_code_155e45b8():
        async def run_async_code_dc1589e3():
            history = await learning_system.get_learning_history()
            return history
        history = asyncio.run(run_async_code_dc1589e3())
        logger.success(format_json(history))
        return history
    history = asyncio.run(run_async_code_155e45b8())
    logger.success(format_json(history))
    logger.debug(history)


if __name__ == "__main__":
    """Run the example"""
    logger.debug("Multi-agent Learning System powered by LlamaIndex and Mem0")

    async def main():
        async def run_async_code_67cc6487():
            await run_learning_agent()
            return 
         = asyncio.run(run_async_code_67cc6487())
        logger.success(format_json())

    asyncio.run(main())

"""
## How It Works

### 1. Memory Context Setup
"""
logger.info("## How It Works")

self.memory_context = {"user_id": student_id, "app": "learning_assistant"}
self.memory = Mem0Memory.from_client(context=self.memory_context)

"""
The memory context identifies the specific student and application, ensuring memory isolation and proper retrieval.

### 2. Agent Collaboration
"""
logger.info("### 2. Agent Collaboration")

can_handoff_to=["PracticeAgent"]  # TutorAgent can hand off to PracticeAgent
can_handoff_to=["TutorAgent"]     # PracticeAgent can hand off back

"""
Agents collaborate seamlessly, with the TutorAgent handling explanations and the PracticeAgent managing exercises.

### 3. Shared Memory
"""
logger.info("### 3. Shared Memory")

async def async_func_0():
    response = await self.workflow.run(
        user_msg=request,
        memory=self.memory  # Shared across all agents
    )
    return response
response = asyncio.run(async_func_0())
logger.success(format_json(response))

"""
All agents in the workflow share the same memory context, enabling true collaborative learning.

### 4. Memory-Driven Interactions

The system prompts guide agents to:
- Reference previous learning sessions
- Adapt to discovered learning styles
- Build progressively on past lessons
- Track and respond to learning patterns

## Running the Example
"""
logger.info("### 4. Memory-Driven Interactions")

learning_system = MultiAgentLearningSystem(student_id="Alexander")

async def async_func_2():
    response = await learning_system.start_learning_session(
        "Vision Language Models",
        "I'm new to machine learning but I have good hold on Python and have 4 years of work experience."
    )
    return response
response = asyncio.run(async_func_2())
logger.success(format_json(response))

async def async_func_7():
    response2 = await learning_system.start_learning_session(
        "Machine Learning",
        "what all did I cover so far?"
    )
    return response2
response2 = asyncio.run(async_func_7())
logger.success(format_json(response2))

async def run_async_code_9f351fe0():
    async def run_async_code_155e45b8():
        history = await learning_system.get_learning_history()
        return history
    history = asyncio.run(run_async_code_155e45b8())
    logger.success(format_json(history))
    return history
history = asyncio.run(run_async_code_9f351fe0())
logger.success(format_json(history))

"""
## Expected Output

The system will demonstrate memory-aware interactions:

Session 1:
I understand you want to learn about Vision Language Models and you mentioned you're new to machine learning but have a strong Python background with 4 years of experience. That's a great foundation to build on!

Let me start with an explanation tailored to your programming background...
[Agent provides explanation and may hand off to PracticeAgent for exercises]

Session 2:
Based on our previous session, I remember we covered Vision Language Models and I noted that you have a strong Python background with 4 years of experience. You mentioned being new to machine learning, so we started with foundational concepts...
[Agent references previous session and builds upon it]

## Key Benefits

1. **Persistent Learning**: Agents remember across sessions, creating continuity
2. **Collaborative Teaching**: Multiple specialized agents work together seamlessly
3. **Personalized Adaptation**: System learns and adapts to individual learning styles
4. **Scalable Architecture**: Easy to add more specialized agents
5. **Memory Efficiency**: Shared memory prevents duplication and ensures consistency


## Best Practices

1. **Clear Agent Roles**: Define specific responsibilities for each agent
2. **Memory Context**: Use descriptive context for memory isolation
3. **Handoff Strategy**: Design clear handoff criteria between agents
5. **Memory Hygiene**: Regularly review and clean memory for optimal performance

## Help & Resources

- [LlamaIndex Agent Workflows](https://docs.llamaindex.ai/en/stable/use_cases/agents/)
- [Mem0 Platform](https://app.mem0.ai/)

<Snippet file="get-help.mdx" />
"""
logger.info("## Expected Output")

logger.info("\n\n[DONE]", bright=True)