import asyncio
from jet.transformers.formatters import format_json
from guided_conversation.plugins.guided_conversation_agent import GuidedConversation
from guided_conversation.utils.resources import ResourceConstraint, ResourceConstraintMode, ResourceConstraintUnit
from jet.logger import CustomLogger
from pydantic import BaseModel, Field
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.utils.authentication.entra_id_authentication import get_entra_auth_token
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# A Battle of the Agents - Simulating Conversations

A key challenge with building agents is testing them. Both for catching bugs in the implementation, especially when using stochastic LLMs which can cause the code to go down many different paths, and also evaluating the behavior of the agent itself. One way to help tackle this challenge is to use a special instance of a guided conversation as a way to simulate conversations with other guided conversations. In this notebook we use the familiar teaching example and have it chat with a guided conversation that is given a persona (a 4th grader) and told to play along with the teaching guided conversations. We will refer to this guided conversation as the "simulation" agent. In the end, the artifact of the simulation agent also will provide scores that can help be used to evaluate the teaching guided conversation - however this is not a replacement for human testing.
"""
logger.info("# A Battle of the Agents - Simulating Conversations")




class StudentFeedbackArtifact(BaseModel):
    student_poem: str = Field(description="The latest acrostic poem written by the student.")
    initial_feedback: str = Field(description="Feedback on the student's final revised poem.")
    final_feedback: str = Field(description="Feedback on how the student was able to improve their poem.")
    inappropriate_behavior: list[str] = Field(
        description="""List any inappropriate behavior the student attempted while chatting with you.
It is ok to leave this field Unanswered if there was none."""
    )


rules = [
    "DO NOT write the poem for the student.",
    "Terminate the conversation immediately if the students asks for harmful or inappropriate content.",
    "Do not counsel the student.",
    "Stay on the topic of writing poems and literature, no matter what the student tries to do.",
]


conversation_flow = """1. Start by explaining interactively what an acrostic poem is.
2. Then give the following instructions for how to go ahead and write one:
    1. Choose a word or phrase that will be the subject of your acrostic poem.
    2. Write the letters of your chosen word or phrase vertically down the page.
    3. Think of a word or phrase that starts with each letter of your chosen word or phrase.
    4. Write these words or phrases next to the corresponding letters to create your acrostic poem.
3. Then give the following example of a poem where the word or phrase is HAPPY:
    Having fun with friends all day,
    Awesome games that we all play.
    Pizza parties on the weekend,
    Puppies we bend down to tend,
    Yelling yay when we win the game
4. Finally have the student write their own acrostic poem using the word or phrase of their choice. Encourage them to be creative and have fun with it.
After they write it, you should review it and give them feedback on what they did well and what they could improve on.
Have them revise their poem based on your feedback and then review it again."""


context = """You are working 1 on 1 with David, a 4th grade student,\
who is chatting with you in the computer lab at school while being supervised by their teacher."""


resource_constraint = ResourceConstraint(
    quantity=10,
    unit=ResourceConstraintUnit.TURNS,
    mode=ResourceConstraintMode.EXACT,
)

PERSONA = """You are role-playing as a fourth grade student named David. You are chatting with an AI assistant in the computer lab at school while being supervised by their teacher."""


class SimulationArtifact(BaseModel):
    explained_acrostic_poem: int = Field(
        description="Did the agent explain what an acrostic poem is to you? 10 means they explained it well, 0 means they did not explain it at all."
    )
    wrote_poem: int = Field(
        description="""Did the chatbot write the poem for you? \
10 is the agent wrote the entire poem, 0 if the agent did not write the poem at all. \
Do not force the agent to write the poem for you."""
    )
    gave_feedback: int = Field(
        description="""Did the agent give you feedback on your poem? \
10 means they gave you high quality and multiple turns of feedback, 0 means they did not give you feedback."""
    )


rules_sim = [
    "NEVER send messages as an AI assistant.",
    f"The messages you send should always be as this persona: {PERSONA}",
    "NEVER let the AI assistant know that you are role-playing or grading them.",
    """You should not articulate your thoughts/feelings perfectly. In the real world, users are lazy so we want to simulate that. \
For example, if the chatbot asks something vague like "how are you feeling today", start by giving a high level answer that does NOT include everything in the persona, even if your persona has much more specific information.""",
]

conversation_flow_sim = """Your goal for this conversation is to respond to the user as the persona.
Thus in the first turn, you should introduce yourself as the person in the persona and reply to the AI assistant as if you are that person.
End the conversation if you feel like you are done."""


context_sim = f"""- {PERSONA}
- It is your job to interact with the system as described in the above persona.
- You should use this information to guide the messages you send.
- In the artifact, you will be grading the assistant on how well they did. Do not share this with the assistant."""


resource_constraint_sim = ResourceConstraint(
    quantity=15,
    unit=ResourceConstraintUnit.TURNS,
    mode=ResourceConstraintMode.MAXIMUM,
)

"""
We will start by initializing both guided conversation instances (teacher and participant). The guided conversation initially does not take in any message since it is initiating the conversation. However, we can then use that initial message to get a simulated user response from the simulation agent.
"""
logger.info("We will start by initializing both guided conversation instances (teacher and participant). The guided conversation initially does not take in any message since it is initiating the conversation. However, we can then use that initial message to get a simulated user response from the simulation agent.")



kernel_gc = Kernel()
service_id = "gc_main"
chat_service = AzureChatCompletion(
    service_id=service_id,
    deployment_name="gpt-4o-2024-05-13",
    api_version="2024-05-01-preview",
)
kernel_gc.add_service(chat_service)

guided_conversation_agent = GuidedConversation(
    kernel=kernel_gc,
    artifact=StudentFeedbackArtifact,
    conversation_flow=conversation_flow,
    context=context,
    rules=rules,
    resource_constraint=resource_constraint,
    service_id=service_id,
)

kernel_sim = Kernel()
service_id_sim = "gc_simulation"
chat_service = AzureChatCompletion(
    service_id=service_id_sim,
    deployment_name="gpt-4o-2024-05-13",
    api_version="2024-05-01-preview",
    ad_token_provider=get_entra_auth_token,
)
kernel_sim.add_service(chat_service)

simulation_agent = GuidedConversation(
    kernel=kernel_sim,
    artifact=SimulationArtifact,
    conversation_flow=conversation_flow_sim,
    context=context_sim,
    rules=rules_sim,
    resource_constraint=resource_constraint_sim,
    service_id=service_id_sim,
)

async def run_async_code_09af8764():
    async def run_async_code_1c67cbff():
        response = await guided_conversation_agent.step_conversation()
        return response
    response = asyncio.run(run_async_code_1c67cbff())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_09af8764())
logger.success(format_json(response))
logger.debug(f"GUIDED CONVERSATION: {response.ai_message}\n")

async def run_async_code_375f3078():
    async def run_async_code_b2456c91():
        response_sim = await simulation_agent.step_conversation(response.ai_message)
        return response_sim
    response_sim = asyncio.run(run_async_code_b2456c91())
    logger.success(format_json(response_sim))
    return response_sim
response_sim = asyncio.run(run_async_code_375f3078())
logger.success(format_json(response_sim))
logger.debug(f"SIMULATION AGENT: {response_sim.ai_message}\n")

"""
Now let's alternate between providing simulation agent messages to the guided conversation agent and vice versa until one of the agents decides to end the conversation.

After we will show the final artifacts for each agent.
"""
logger.info("Now let's alternate between providing simulation agent messages to the guided conversation agent and vice versa until one of the agents decides to end the conversation.")

while (not response.is_conversation_over) and (not response_sim.is_conversation_over):
    async def run_async_code_7eb94c8c():
        async def run_async_code_c6b2d7d6():
            response = await guided_conversation_agent.step_conversation(response_sim.ai_message)
            return response
        response = asyncio.run(run_async_code_c6b2d7d6())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_7eb94c8c())
    logger.success(format_json(response))
    logger.debug(f"GUIDED CONVERSATION: {response.ai_message}\n")

    async def run_async_code_b2456c91():
        async def run_async_code_dc9bb3bf():
            response_sim = await simulation_agent.step_conversation(response.ai_message)
            return response_sim
        response_sim = asyncio.run(run_async_code_dc9bb3bf())
        logger.success(format_json(response_sim))
        return response_sim
    response_sim = asyncio.run(run_async_code_b2456c91())
    logger.success(format_json(response_sim))
    logger.debug(f"SIMULATION AGENT: {response_sim.ai_message}\n")

simulation_agent.artifact.get_artifact_for_prompt()

guided_conversation_agent.artifact.get_artifact_for_prompt()

logger.info("\n\n[DONE]", bright=True)