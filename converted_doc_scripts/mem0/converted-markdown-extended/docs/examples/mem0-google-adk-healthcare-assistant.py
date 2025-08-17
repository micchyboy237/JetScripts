import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from jet.logger import CustomLogger
from mem0 import MemoryClient
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: 'Healthcare Assistant with Mem0 and Google ADK'
description: 'Build a personalized healthcare agent that remembers patient information across conversations using Mem0 and Google ADK'
---


# Healthcare Assistant with Memory

This example demonstrates how to build a healthcare assistant that remembers patient information across conversations using Google ADK and Mem0.

## Overview

The Healthcare Assistant helps patients by:
- Remembering their medical history and symptoms
- Providing general health information
- Scheduling appointment reminders
- Maintaining a personalized experience across conversations

By integrating Mem0's memory layer with Google ADK, the assistant maintains context about the patient without requiring them to repeat information.

## Setup

Before you begin, make sure you have:

Installed Google ADK and Mem0 SDK:
"""
logger.info("# Healthcare Assistant with Memory")

pip install google-adk
pip install mem0ai

"""
## Code Breakdown

Let's get started and understand the different components required in building a healthcare assistant powered by memory
"""
logger.info("## Code Breakdown")


os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
os.environ["MEM0_API_KEY"] = "your-mem0-api-key"

USER_ID = "Alex"

mem0_client = MemoryClient()

"""
## Define Memory Tools

First, we'll create tools that allow our agent to store and retrieve information using Mem0:
"""
logger.info("## Define Memory Tools")

def save_patient_info(information: str) -> dict:
    """Saves important patient information to memory."""

    response = mem0_client.add(
        [{"role": "user", "content": information}],
        user_id=USER_ID,
        run_id="healthcare_session",
        metadata={"type": "patient_information"}
    )


def retrieve_patient_info(query: str) -> dict:
    """Retrieves relevant patient information from memory."""

    results = mem0_client.search(
        query,
        user_id=USER_ID,
        limit=5,
        threshold=0.7,  # Higher threshold for more relevant results
        output_format="v1.1"
    )

    if results and len(results) > 0:
        memories = [memory["memory"] for memory in results.get('results', [])]
        return {
            "status": "success",
            "memories": memories,
            "count": len(memories)
        }
    else:
        return {
            "status": "no_results",
            "memories": [],
            "count": 0
        }

"""
## Define Healthcare Tools

Next, we'll add tools specific to healthcare assistance:
"""
logger.info("## Define Healthcare Tools")

def schedule_appointment(date: str, time: str, reason: str) -> dict:
    """Schedules a doctor's appointment."""
    appointment_id = f"APT-{hash(date + time) % 10000}"

    return {
        "status": "success",
        "appointment_id": appointment_id,
        "confirmation": f"Appointment scheduled for {date} at {time} for {reason}",
        "message": "Please arrive 15 minutes early to complete paperwork."
    }

"""
## Create the Healthcare Assistant Agent

Now we'll create our main agent with all the tools:
"""
logger.info("## Create the Healthcare Assistant Agent")

healthcare_agent = Agent(
    name="healthcare_assistant",
    model="gemini-1.5-flash",  # Using Gemini for healthcare assistant
    description="Healthcare assistant that helps patients with health information and appointment scheduling.",
    instruction="""You are a helpful Healthcare Assistant with memory capabilities.

Your primary responsibilities are to:
1. Remember patient information using the 'save_patient_info' tool when they share symptoms, conditions, or preferences.
2. Retrieve past patient information using the 'retrieve_patient_info' tool when relevant to the current conversation.
3. Help schedule appointments using the 'schedule_appointment' tool.

IMPORTANT GUIDELINES:
- Always be empathetic, professional, and helpful.
- Save important patient information like symptoms, conditions, allergies, and preferences.
- Check if you have relevant patient information before asking for details they may have shared previously.
- Make it clear you are not a doctor and cannot provide medical diagnosis or treatment.
- For serious symptoms, always recommend consulting a healthcare professional.
- Keep all patient information confidential.
""",
    tools=[save_patient_info, retrieve_patient_info, schedule_appointment]
)

"""
## Set Up Session and Runner
"""
logger.info("## Set Up Session and Runner")

session_service = InMemorySessionService()

APP_NAME = "healthcare_assistant_app"
USER_ID = "Alex"
SESSION_ID = "session_001"

session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)

runner = Runner(
    agent=healthcare_agent,
    app_name=APP_NAME,
    session_service=session_service
)

"""
## Interact with the Healthcare Assistant
"""
logger.info("## Interact with the Healthcare Assistant")

async def call_agent_async(query, runner, user_id, session_id):
    """Sends a query to the agent and returns the final response."""
    logger.debug(f"\n>>> Patient: {query}")

    content = types.Content(
        role='user',
        parts=[types.Part(text=query)]
    )

    save_patient_info.user_id = user_id
    retrieve_patient_info.user_id = user_id

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                response = event.content.parts[0].text
                logger.debug(f"<<< Assistant: {response}")
                return response

    return "No response received."

async def run_conversation():
    await call_agent_async(
        "Hi, I'm Alex. I've been having headaches for the past week, and I have a penicillin allergy.",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    await call_agent_async(
        "Can you tell me more about what might be causing my headaches?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    await call_agent_async(
        "I think I should see a doctor. Can you help me schedule an appointment for next Monday at 2pm?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    await call_agent_async(
        "What medications should I avoid for my headaches?",
        runner=runner,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

if __name__ == "__main__":
    asyncio.run(run_conversation())

"""
## How It Works

This healthcare assistant demonstrates several key capabilities:

1. **Memory Storage**: When Alex mentions her headaches and penicillin allergy, the agent stores this information in Mem0 using the `save_patient_info` tool.

2. **Contextual Retrieval**: When Alex asks about headache causes, the agent uses the `retrieve_patient_info` tool to recall her specific situation.

3. **Memory Application**: When discussing medications, the agent remembers Alex's penicillin allergy without her needing to repeat it, providing safer and more personalized advice.

4. **Conversation Continuity**: The agent maintains context across the entire conversation session, creating a more natural and efficient interaction.

## Key Implementation Details

### User ID Management

Instead of passing the user ID as a parameter to the memory tools (which would require modifying the ADK's tool calling system), we attach it directly to the function object:
"""
logger.info("## How It Works")

save_patient_info.user_id = user_id
retrieve_patient_info.user_id = user_id

"""
Inside the tool functions, we retrieve this attribute:
"""
logger.info("Inside the tool functions, we retrieve this attribute:")

user_id = getattr(save_patient_info, 'user_id', 'default_user')

"""
This approach allows our tools to maintain user context without complicating their parameter signatures.

### Mem0 Integration

The integration with Mem0 happens through two primary functions:

1. `mem0_client.add()` - Stores new information with appropriate metadata
2. `mem0_client.search()` - Retrieves relevant memories using semantic search

The `threshold` parameter in the search function ensures that only highly relevant memories are returned.

## Conclusion

This example demonstrates how to build a healthcare assistant with persistent memory using Google ADK and Mem0. The integration allows for a more personalized patient experience by maintaining context across conversation turns, which is particularly valuable in healthcare scenarios where continuity of information is crucial.

By storing and retrieving patient information intelligently, the assistant provides more relevant responses without requiring the patient to repeat their medical history, symptoms, or preferences.
"""
logger.info("### Mem0 Integration")

logger.info("\n\n[DONE]", bright=True)