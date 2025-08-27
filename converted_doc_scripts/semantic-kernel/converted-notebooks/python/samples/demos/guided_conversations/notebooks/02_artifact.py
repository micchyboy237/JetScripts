import asyncio
from jet.transformers.formatters import format_json
from guided_conversation.plugins.artifact import Artifact
from guided_conversation.utils.conversation_helpers import Conversation
from jet.logger import CustomLogger
from pydantic import BaseModel, Field, conlist
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import AuthorRole, ChatMessageContent
from typing import Literal
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# The Guided Conversation Artifact
This notebook explores one of our core modular components or plugins, the Artifact.

The artifact is a form, or a type of working memory for the agent. We implement it using a Pydantic BaseModel. As the conversation creator, you can define an arbitrary BaseModel that includes the fields you want the agent to fill out during the conversation.

## Motivating Example - Collecting Information from a User

Let's setup an artifact where the goal is to collect information about a customer's issue with a service.
"""
logger.info("# The Guided Conversation Artifact")




class Issue(BaseModel):
    incident_type: Literal["Service Outage", "Degradation", "Billing", "Security", "Data Loss", "Other"] = Field(
        description="A high level type describing the incident."
    )
    description: str = Field(description="A detailed description of what is going wrong.")
    affected_services: conlist(str, min_length=0) = Field(description="The services affected by the incident.")


class OutageArtifact(BaseModel):
    name: str = Field(description="How to address the customer.")
    company: str = Field(description="The company the customer works for.")
    role: str = Field(description="The role of the customer.")
    email: str = Field(description="The best email to contact the customer.", pattern=r"^/^.+@.+$/$")
    phone: str = Field(description="The best phone number to contact the customer.", pattern=r"^\d{3}-\d{3}-\d{4}$")

    incident_start: int = Field(
        description="About how many hours ago the incident started.",
    )
    incident_end: int = Field(
        description="About how many hours ago the incident ended. If the incident is ongoing, set this to 0.",
    )

    issues: conlist(Issue, min_length=1) = Field(description="The issues the customer is experiencing.")
    additional_comments: conlist(str, min_length=0) = Field("Any additional comments the customer has.")

"""
Let's initialize the artifact as a standalone module.

It requires a Kernel and LLM Service, alongside a Conversation object.
"""
logger.info("Let's initialize the artifact as a standalone module.")



kernel = Kernel()
service_id = "artifact_chat_completion"
chat_service = AzureChatCompletion(
    service_id=service_id,
    deployment_name="gpt-4o-2024-05-13",
    api_version="2024-05-01-preview",
)
kernel.add_service(chat_service)

artifact = Artifact(kernel, service_id, OutageArtifact, max_artifact_field_retries=2)
conversation = Conversation()

"""
To power the Artifact's ability to automatically fix issues, we provide the conversation history as additional context.
"""
logger.info("To power the Artifact's ability to automatically fix issues, we provide the conversation history as additional context.")


conversation.add_messages(
    ChatMessageContent(
        role=AuthorRole.ASSISTANT,
        content="Hello! I'm here to help you with your issue. Can you tell me your name, company, and role?",
    )
)
conversation.add_messages(
    ChatMessageContent(
        role=AuthorRole.USER,
        content="Yes my name is Jane Doe, I work at Contoso, and I'm a database uhh administrator.",
    )
)

async def async_func_15():
    result = await artifact.update_artifact(
        field_name="name",
        field_value="Jane Doe",
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_15())
logger.success(format_json(result))
conversation.add_messages(result.messages)

async def async_func_22():
    result = await artifact.update_artifact(
        field_name="company",
        field_value="Contoso",
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_22())
logger.success(format_json(result))
conversation.add_messages(result.messages)

async def async_func_29():
    result = await artifact.update_artifact(
        field_name="role",
        field_value="Database Administrator",
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_29())
logger.success(format_json(result))
conversation.add_messages(result.messages)

"""
Let's see how the artifact was updated with these valid updates and the resulting conversation messages that were generated.

The Artifact creates messages whenever a field is updated for use in downstream agents like the main GuidedConversation.
"""
logger.info("Let's see how the artifact was updated with these valid updates and the resulting conversation messages that were generated.")

logger.debug(f"Conversation up to this point:\n{conversation.get_repr_for_prompt()}\n")
logger.debug(f"Current state of the artifact:\n{artifact.get_artifact_for_prompt()}")

"""
Next we test an invalid update on a field with a regex. The agent should not update the artifact and
instead resume the conversation because the provided email is incomplete.
"""
logger.info("Next we test an invalid update on a field with a regex. The agent should not update the artifact and")

conversation.add_messages(
    ChatMessageContent(role=AuthorRole.ASSISTANT, content="What is the best email to contact you at?")
)
conversation.add_messages(ChatMessageContent(role=AuthorRole.USER, content="my email is jdoe"))
async def async_func_4():
    result = await artifact.update_artifact(
        field_name="email",
        field_value="jdoe",
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_4())
logger.success(format_json(result))
conversation.add_messages(result.messages)

"""
If the agent returned success, but did make an update (as shown by not generating a conversation message indicating such),
then we implicitly assume the agent has resumed the conversation.
"""
logger.info("If the agent returned success, but did make an update (as shown by not generating a conversation message indicating such),")

logger.debug(f"Conversation up to this point:\n{conversation.get_repr_for_prompt()}")

"""
Now let's see what happens if we keep trying to update that failed field.
"""
logger.info("Now let's see what happens if we keep trying to update that failed field.")

async def async_func_0():
    result = await artifact.update_artifact(
        field_name="email",
        field_value="jdoe",
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_0())
logger.success(format_json(result))

async def async_func_6():
    result = await artifact.update_artifact(
        field_name="email",
        field_value="jdoe",
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_6())
logger.success(format_json(result))

"""
If we look at the current state of the artifact, we should see that the email has been removed
since it has now failed 3 times which is greater than the max_artifact_field_retries parameter we set
when we instantiated the artifact.
"""
logger.info("If we look at the current state of the artifact, we should see that the email has been removed")

artifact.get_artifact_for_prompt()

"""
Now let's move on to trying to update a more complex field: the issues field.
"""
logger.info("Now let's move on to trying to update a more complex field: the issues field.")

conversation.add_messages(
    ChatMessageContent(role=AuthorRole.ASSISTANT, content="Can you tell me about the issues you're experiencing?")
)
conversation.add_messages(
    ChatMessageContent(
        role=AuthorRole.USER,
        content="""The latency of accessing our database service has increased by 200\% in the last 24 hours,
even on a fresh instance. Additionally, we're seeing a lot of timeouts when trying to access the management portal.""",
    )
)

async def async_func_11():
    result = await artifact.update_artifact(
        field_name="issues",
        field_value=[
            {
                "incident_type": "Degradation",
                "description": """The latency of accessing the customer's database service has increased by 200% in the \
    last 24 hours, even on a fresh instance. They also report timeouts when trying to access the management portal.""",
                "affected_services": ["Database Service", "Database Management Portal"],
            }
        ],
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_11())
logger.success(format_json(result))
conversation.add_messages(result.messages)

logger.debug(f"Conversation up to this point:\n{conversation.get_repr_for_prompt()}\n")
logger.debug(f"Current state of the artifact:\n{artifact.get_artifact_for_prompt()}")

"""
To add another affected service, we can need to update the issues field with the new value again.
The obvious con of this approach is that the model generating the field_value has to regenerate the entire field_value.
However, the pro is that keeps the available tools simple for the model.
"""
logger.info("To add another affected service, we can need to update the issues field with the new value again.")

conversation.add_messages(
    ChatMessageContent(
        role=AuthorRole.ASSISTANT,
        content="Is there anything else you'd like to add about the issues you're experiencing?",
    )
)
conversation.add_messages(
    ChatMessageContent(
        role=AuthorRole.USER,
        content="Yes another thing that is effected is access to billing information is very slow.",
    )
)

async def async_func_13():
    result = await artifact.update_artifact(
        field_name="issues",
        field_value=[
            {
                "incident_type": "Degradation",
                "description": """The latency of accessing the customer's database service has increased by 200% in the \
    last 24 hours, even on a fresh instance. They also report timeouts when trying to access the \
    management portal and slowdowns in the access to billing information.""",
                "affected_services": ["Database Service", "Database Management Portal", "Billing portal"],
            },
        ],
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_13())
logger.success(format_json(result))
conversation.add_messages(result.messages)
logger.debug(f"Conversation up to this point:\n{conversation.get_repr_for_prompt()}\n")
logger.debug(f"Current state of the artifact:\n{artifact.get_artifact_for_prompt()}")

"""
Now let's see what happens if we try to update a field that is not in the artifact.
"""
logger.info("Now let's see what happens if we try to update a field that is not in the artifact.")

async def async_func_0():
    result = await artifact.update_artifact(
        field_name="not_a_field",
        field_value="some value",
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_0())
logger.success(format_json(result))
logger.debug(f"Was the update successful? {result.update_successful}")
logger.debug(f"Conversation up to this point:\n{conversation.get_repr_for_prompt()}\n")
logger.debug(f"Current state of the artifact:\n{artifact.get_artifact_for_prompt()}")

"""
Finally, let's see what happens if we try to update a field with the incorrect type, but the correct information was provided in the conversation. 
We should see the agent correctly updated the field correctly as an integer.
"""
logger.info("Finally, let's see what happens if we try to update a field with the incorrect type, but the correct information was provided in the conversation.")

conversation.add_messages(
    ChatMessageContent(role=AuthorRole.ASSISTANT, content="How many hours ago did the incident start?")
)
conversation.add_messages(ChatMessageContent(role=AuthorRole.USER, content="about 3 hours ago"))
async def async_func_4():
    result = await artifact.update_artifact(
        field_name="incident_start",
        field_value="3 hours",
        conversation=conversation,
    )
    return result
result = asyncio.run(async_func_4())
logger.success(format_json(result))

logger.debug(f"Current state of the artifact:\n{artifact.get_artifact_for_prompt()}")

logger.info("\n\n[DONE]", bright=True)