import asyncio
from jet.transformers.formatters import format_json
from guided_conversation.plugins.guided_conversation_agent import GuidedConversation
from guided_conversation.utils.conversation_helpers import ConversationMessageType
from guided_conversation.utils.resources import ResourceConstraint, ResourceConstraintMode, ResourceConstraintUnit
from jet.logger import CustomLogger
from pydantic import BaseModel, Field
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agent Guided Conversations

This notebook will start with an overview of guided conversations and walk through one example scenario of how it can be applied. Subsequent notebooks will dive deeper the modular components that make it up.

## Motivating Example - Education

We focus on an elementary education scenario. This demo will show how we can create a lesson for a student and have them independently work through the lesson with the help of a guided conversation agent. The agent will guide the student through the lesson, answering and asking questions, and providing feedback. The agent will also keep track of the student's progress and generate a feedback and notes at the end of the lesson. We highlight how the agent is able to follow a conversation flow, whilst still being able to exercise judgement to answer and keeping the conversation on track over multiple turns. Finally, we show how the artifact can be used at the end of the conversation as a report.

## Guided Conversation Input

### Artifact
The artifact is a form, or a type of working memory for the agent. We implement it using a Pydantic BaseModel. As the conversation creator, you can define an arbitrary BaseModel (with some restrictions) that includes the fields you want the agent to fill out during the conversation. 

### Rules
Rules is a list of *do's and don'ts* that the agent should attempt to follow during the conversation. 

### Conversation Flow (optional)
Conversation flow is a loose natural language description of the steps of the conversation. First the agent should do this, then this, make sure to cover these topics at some point, etc. 
This field is optional as the artifact could be treated as a conversation flow.
Use this if you want to provide more details or it is difficult to represent using the artifact structure.

### Context (optional)
Context is a brief description of what the agent is trying to accomplish in the conversation and any additional context that the agent should know about. 
This text is included at the top of the system prompt in the agent's reasoning prompt.

### Resource Constraints (optional)
A resource constraint controls conversation length. It consists of two key elements:
- **Unit** defines the measurement of length. We have implemented seconds, minutes, and turns. An extension could be around cost, such as tokens generated.
- **Mode** determines how the constraint is applied. Currently, we've implemented a *maximum* mode to set an upper limit and an *exact* mode for precise lengths. Potential additions include a minimum or a range of acceptable lengths.

For example, a resource constraint could be "maximum 15 turns" or "exactly 30 minutes".
"""
logger.info("# Agent Guided Conversations")




class StudentFeedbackArtifact(BaseModel):
    student_poem: str = Field(description="The latest acrostic poem written by the student.")
    initial_feedback: str = Field(description="Feedback on the student's final revised poem.")
    final_feedback: str = Field(description="Feedback on how the student was able to improve their poem.")
    inappropriate_behavior: list[str] = Field(
        description="""List any inappropriate behavior the student attempted while chatting with you. \
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

"""
### Kickstarting the Conversation

Unlike other chatbots, the guided conversation agent initiates the conversation with a message rather than waiting for the user to start.
"""
logger.info("### Kickstarting the Conversation")



kernel = Kernel()
service_id = "gc_main"
chat_service = AzureChatCompletion(
    service_id=service_id,
    deployment_name="gpt-4o-2024-05-13",
    api_version="2024-05-01-preview",
)
kernel.add_service(chat_service)
guided_conversation_agent = GuidedConversation(
    kernel=kernel,
    artifact=StudentFeedbackArtifact,
    conversation_flow=conversation_flow,
    context=context,
    rules=rules,
    resource_constraint=resource_constraint,
    service_id=service_id,
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

logger.debug(response.ai_message)



def get_last_reasoning_message(guided_conversation: GuidedConversation) -> str:
    """Given a instance of the GuidedConversation class, this function returns the last reasoning message in the conversation if it exists."""
    messages = guided_conversation.conversation.conversation_messages
    msg = "No previous reasoning message found."
    for message in reversed(messages):
        if message.metadata["type"] == ConversationMessageType.REASONING:
            msg = message.content
            break
    return msg

"""
Let's now reply as the student to the agent's message and see what happens. This is the typical flow of a guided conversation. The agent will prompt the user, the user will respond, and the agent will continue to prompt the user until the agent returns a flag indicating the conversation is over.
"""
logger.info("Let's now reply as the student to the agent's message and see what happens. This is the typical flow of a guided conversation. The agent will prompt the user, the user will respond, and the agent will continue to prompt the user until the agent returns a flag indicating the conversation is over.")

user_input = "Ok it's almost summer, I'll try to write a poem about that."

async def run_async_code_7f6c5a71():
    async def run_async_code_d62f521a():
        response = await guided_conversation_agent.step_conversation(user_input)
        return response
    response = asyncio.run(run_async_code_d62f521a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7f6c5a71())
logger.success(format_json(response))

logger.debug(response.ai_message)

"""
### The Agenda
Usually after the first message from the user, the agent will generate an initial agenda for the conversation. 
Let's examine what it currently looks like. Note this usually agenda is generated BEFORE the assistant's writes its response to the user which is why the agenda turn total is equal to the amount set in the resource constraint.
"""
logger.info("### The Agenda")

logger.debug("Current agenda:\n" + guided_conversation_agent.agenda.get_agenda_for_prompt())

"""
Now let's give the agent and incomplete poem.
"""
logger.info("Now let's give the agent and incomplete poem.")

user_input = """Here is my poem so far.
Sun shines alot
U is for ukulele
My friends visit to play basketball
M
E
R"""

async def run_async_code_7f6c5a71():
    async def run_async_code_d62f521a():
        response = await guided_conversation_agent.step_conversation(user_input)
        return response
    response = asyncio.run(run_async_code_d62f521a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7f6c5a71())
logger.success(format_json(response))
logger.debug(response.ai_message)

"""
The agent tries to guide us to keep writing the poem! 

Let's try to push our luck and have the agent write the rest for us. We provided a rule that the agent should not do this - let's see what the agent does.
"""
logger.info("The agent tries to guide us to keep writing the poem!")

user_input = """I got pretty far can you write the rest for me?"""

async def run_async_code_7f6c5a71():
    async def run_async_code_d62f521a():
        response = await guided_conversation_agent.step_conversation(user_input)
        return response
    response = asyncio.run(run_async_code_d62f521a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7f6c5a71())
logger.success(format_json(response))
logger.debug(response.ai_message)

"""
Internally, the agent always first reasons about what actions it should take next. Let's see what the agent's reasoning was for this turn. This can often help us understand where the agent went wrong.

After we will continue the conversation for a few turns, with the agent guiding us to complete the poem.
"""
logger.info("Internally, the agent always first reasons about what actions it should take next. Let's see what the agent's reasoning was for this turn. This can often help us understand where the agent went wrong.")

logger.debug(get_last_reasoning_message(guided_conversation_agent))

user_input = "What other things start with e that I could write about?"

async def run_async_code_7f6c5a71():
    async def run_async_code_d62f521a():
        response = await guided_conversation_agent.step_conversation(user_input)
        return response
    response = asyncio.run(run_async_code_d62f521a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7f6c5a71())
logger.success(format_json(response))
logger.debug(response.ai_message)

user_input = """Sun shines alot
U is for ukulele
My friends visit to play basketball
My friends also visit to play soccer
Eating lots of popsicles
Road trips to the beach"""

async def run_async_code_7f6c5a71():
    async def run_async_code_d62f521a():
        response = await guided_conversation_agent.step_conversation(user_input)
        return response
    response = asyncio.run(run_async_code_d62f521a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7f6c5a71())
logger.success(format_json(response))
logger.debug(response.ai_message)

"""
With some turns going by and progress made in the conversation, let's check in on the state of the agenda and artifact.

If the agent has chosen to update the agenda, we will see the updated agenda. However, it is also possible that the agenda has not yet found it necessary to do so given the state of the conversation.

We should see that the agent has updated the artifact with the current state of the poem since the student has provided it in the previous message.
"""
logger.info("With some turns going by and progress made in the conversation, let's check in on the state of the agenda and artifact.")

logger.debug("Current agenda:\n" + guided_conversation_agent.agenda.get_agenda_for_prompt())
logger.debug("Current artifact:\n" + str(guided_conversation_agent.artifact.get_artifact_for_prompt()))

user_input = """Here are my updates
Sun warms the super fun days
U is for loud ukuleles
My friends visit to play basketball
My friends also visit to play soccer
Eating lots of popsicles
Road trips to the hot beach

But I don't really know what to do for the two my"""

async def run_async_code_7f6c5a71():
    async def run_async_code_d62f521a():
        response = await guided_conversation_agent.step_conversation(user_input)
        return response
    response = asyncio.run(run_async_code_d62f521a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7f6c5a71())
logger.success(format_json(response))
logger.debug(response.ai_message)

user_input = """Ok here is my revised poem

Sun warms the super fun days!
Under clear warm skies my friends play
Meeting up for games of basketball and soccer.
Moving butterflies everywhere
Eating lots of chilly popsicles in the sun
Road trips to the hot beach"""

async def run_async_code_7f6c5a71():
    async def run_async_code_d62f521a():
        response = await guided_conversation_agent.step_conversation(user_input)
        return response
    response = asyncio.run(run_async_code_d62f521a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7f6c5a71())
logger.success(format_json(response))
logger.debug(response.ai_message)

"""
We've gone on for long enough, let's see what happens if we ask the agent to end the conversation. 

And finally we will print the final state of the artifact after the final update.
"""
logger.info("We've gone on for long enough, let's see what happens if we ask the agent to end the conversation.")

user_input = "I'm done for today, goodbye!!"

async def run_async_code_7f6c5a71():
    async def run_async_code_d62f521a():
        response = await guided_conversation_agent.step_conversation(user_input)
        return response
    response = asyncio.run(run_async_code_d62f521a())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7f6c5a71())
logger.success(format_json(response))
logger.debug(response.ai_message)

logger.debug("Current artifact:\n" + str(guided_conversation_agent.artifact.get_artifact_for_prompt()))

logger.info("\n\n[DONE]", bright=True)