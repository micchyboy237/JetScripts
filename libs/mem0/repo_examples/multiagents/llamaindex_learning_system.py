"""
Multi-Agent Personal Learning System: Mem0 + LlamaIndex AgentWorkflow Example
INSTALLATIONS:
!pip install llama-index-core llama-index-memory-mem0 openai
You need MEM0_API_KEY and OPENAI_API_KEY to run the example.
"""

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from jet.adapters.llama_index.factory import get_llama_cpp_llm, get_mem0_local_memory
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    AgentStream,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.tools import FunctionTool

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR / "learning_system.log")),
    ],
)
logger = logging.getLogger(__name__)


class MultiAgentLearningSystem:
    """
    Multi-Agent Architecture:
    - TutorAgent: Main teaching and explanations
    - PracticeAgent: Exercises and skill reinforcement
    - Shared Memory: Both agents learn from student interactions
    """

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.llm = get_llama_cpp_llm()
        self.memory_context = {"user_id": student_id, "app": "learning_assistant"}
        self.memory = get_mem0_local_memory(user_id=student_id)
        self._setup_agents()

    def _setup_agents(self):
        """Setup two agents that work together and share memory"""

        async def assess_understanding(topic: str, student_response: str) -> str:
            """Assess student's understanding of a topic and save insights"""
            if (
                "confused" in student_response.lower()
                or "don't understand" in student_response.lower()
            ):
                assessment = f"STRUGGLING with {topic}: {student_response}"
                insight = f"Student needs more help with {topic}. Prefers step-by-step explanations."
            elif (
                "makes sense" in student_response.lower()
                or "got it" in student_response.lower()
            ):
                assessment = f"UNDERSTANDS {topic}: {student_response}"
                insight = (
                    f"Student grasped {topic} quickly. Can move to advanced concepts."
                )
            else:
                assessment = f"PARTIAL understanding of {topic}: {student_response}"
                insight = (
                    f"Student has basic understanding of {topic}. Needs reinforcement."
                )
            return f"Assessment: {assessment}\nInsight saved: {insight}"

        async def track_progress(topic: str, success_rate: str) -> str:
            """Track learning progress and identify patterns"""
            progress_note = f"Progress on {topic}: {success_rate} - {datetime.now().strftime('%Y-%m-%d')}"
            return f"Progress tracked: {progress_note}"

        tools = [
            FunctionTool.from_defaults(async_fn=assess_understanding),
            FunctionTool.from_defaults(async_fn=track_progress),
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
            can_handoff_to=["PracticeAgent"],
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
            can_handoff_to=["TutorAgent"],
        )

        self.workflow = AgentWorkflow(
            agents=[self.tutor_agent, self.practice_agent],
            root_agent=self.tutor_agent.name,
            initial_state={
                "current_topic": "",
                "student_level": "beginner",
                "learning_style": "unknown",
                "session_goals": [],
            },
        )

    async def start_learning_session(
        self, topic: str, student_message: str = ""
    ) -> str:
        """
        Start a learning session with multi-agent memory-aware teaching.

        Streams the LLM output as it's generated instead of waiting for the
        full response. Each streamed chunk (delta) is logged immediately with
        flush=True so output is visible in real time in both console and log
        file, rather than buffering until the full response completes.

        Also logs timing around workflow start and the first received event,
        so slow steps (e.g. mem0's first-call model loading inside
        SafeMem0Memory.get()) are visible instead of looking like a hang.
        """
        import time

        if student_message:
            request = f"I want to learn about {topic}. {student_message}"
        else:
            request = f"I want to learn about {topic}."

        logger.info(f"Starting session for topic='{topic}'")
        logger.info(f"Request: {request}")

        t_start = time.time()
        logger.info("Calling workflow.run() — awaiting first event...")

        handler = self.workflow.run(user_msg=request, memory=self.memory)

        current_agent = None
        full_response = ""
        chunk_count = 0
        first_event_logged = False

        async for event in handler.stream_events():
            if not first_event_logged:
                logger.info(
                    f"First event received after {time.time() - t_start:.2f}s: "
                    f"{type(event).__name__}"
                )
                first_event_logged = True

            if isinstance(event, AgentInput):
                current_agent = event.current_agent_name
                logger.info(f"[{current_agent}] received input")

            elif isinstance(event, AgentStream):
                # Read the agent name directly off the event itself — this is
                # exactly how function_agent.py / multi_agent_workflow.py set
                # it (current_agent_name=self.name at emit time), so it's
                # always correct per-chunk even across TutorAgent <-> PracticeAgent
                # handoffs, with no dependency on event-ordering assumptions.
                agent_name = event.current_agent_name or current_agent or "agent"
                chunk_count += 1
                full_response += event.delta
                print(event.delta, end="", flush=True)
                # INFO (not DEBUG) — logging.basicConfig is set to INFO, so
                # DEBUG records were silently dropped and never reached
                # console or the log file. INFO guarantees every chunk is
                # actually recorded for traceability.
                logger.info(f"[{agent_name}] chunk#{chunk_count}: {event.delta!r}")

            elif isinstance(event, ToolCall):
                logger.info(
                    f"[{current_agent}] tool_call -> {event.tool_name}({event.tool_kwargs})"
                )

            elif isinstance(event, ToolCallResult):
                logger.info(
                    f"[{current_agent}] tool_result <- {event.tool_name}: {event.tool_output}"
                )

            elif isinstance(event, AgentOutput):
                current_agent = event.current_agent_name or current_agent
                logger.info(f"[{current_agent}] finished output")

        # Trailing newline so subsequent log lines don't run into the last
        # streamed chunk on the console.
        print(flush=True)

        response = await handler
        logger.info(
            f"Session complete for topic='{topic}' — {chunk_count} chunk(s) streamed, "
            f"total_time={time.time() - t_start:.2f}s, "
            f"final_response_len={len(str(response))}"
        )
        return str(response)

    async def get_learning_history(self) -> str:
        """Show what the system remembers about this student"""
        try:
            memories = self.memory.search(
                user_id=self.student_id, query="learning machine learning"
            )
            if memories and len(memories):
                history = "\n".join(f"- {m['memory']}" for m in memories)
                return history
            else:
                return (
                    "No learning history found yet. Let's start building your profile!"
                )
        except Exception as e:
            return f"Memory retrieval error: {str(e)}"


async def run_learning_agent():
    learning_system = MultiAgentLearningSystem(student_id="Alexander")

    logger.info("Session 1:")
    response = await learning_system.start_learning_session(
        "Vision Language Models",
        "I'm new to machine learning but I have good hold on Python and have 4 years of work experience.",
    )
    logger.info(response)

    logger.info("\nSession 2:")
    response2 = await learning_system.start_learning_session(
        "Machine Learning", "what all did I cover so far?"
    )
    logger.info(response2)

    logger.info("\nLearning History:")
    history = await learning_system.get_learning_history()
    logger.info(history)


if __name__ == "__main__":
    """Run the example"""
    logger.info("Multi-agent Learning System powered by LlamaIndex and Mem0")

    async def main():
        await run_learning_agent()

    asyncio.run(main())
