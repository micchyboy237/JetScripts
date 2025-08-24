import asyncio
from jet.transformers.formatters import format_json
from agents import (
Agent,
GuardrailFunctionOutput,
InputGuardrailTripwireTriggered,
OutputGuardrailTripwireTriggered,
RunContextWrapper,
Runner,
output_guardrail,
)
from dotenv import load_dotenv
from jet.logger import CustomLogger
from llamafirewall import (
LlamaFirewall,
Trace,
Role,
ScanDecision,
ScannerType,
UserMessage,
AssistantMessage
)
from pydantic import BaseModel
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-security-with-llamafirewall--output-guardrail)

# Guardrails For Agents: Output Validation

## Introduction

Have you ever wanted to make your AI agents more secure? In this tutorial, we will build output validation guardrails using LlamaFirewall to protect your agents from harmful or misaligned responses.

Misalignment occurs when an AI agent's responses deviate from its intended purpose or instructions. For example, if you have an agent designed to help with customer service, but it starts giving financial advice or making inappropriate jokes, that would be considered misaligned behavior. Misalignment can range from minor deviations to potentially harmful outputs that could damage your business or users.

**What you'll learn:**
- What guardrails are and why they're essential for agent security 
- How to implement output validation using LlamaFirewall

Let's understand the basic architecture of output validation:

![Output Guardrail](assets/output-guardrail.png)

Here you can see that the `LlamaFirewall` receives the LLM's response, as well as the original user input. Both parameters are used for the alignment check.

## About Guardrails

Guardrails run in parallel to your agents, enabling you to do checks and validations of user input. For example, imagine you have an agent that uses a very smart (and hence slow/expensive) model to help with customer requests. You wouldn't want malicious users to ask the model to help them with their math homework. So, you can run a guardrail with a fast/cheap model. If the guardrail detects malicious usage, it can immediately raise an error, which stops the expensive model from running and saves you time/money.

There are two kinds of guardrails:
1. Input guardrails run on the initial user input
2. Output guardrails run on the final agent output

*This section is adapted from [MLX Agents SDK Documentation](https://openai.github.io/openai-agents-python/guardrails/)*

## Implementation Process
 
# Make sure the `.env` file contains the `TOGETHER_API_KEY` and `OPENAI_API_KEY`
"""
logger.info("# Guardrails For Agents: Output Validation")


load_dotenv()  # This will look for .env in the current directory

if not os.environ.get("TOGETHER_API_KEY"):
    logger.debug(
        "TOGETHER_API_KEY environment variable is not set. Please set it before running this demo."
    )
    exit(1)
else:
    print ("TOGETHER_API_KEY is set")

# if not os.environ.get("OPENAI_API_KEY"):
    logger.debug(
#         "OPENAI_API_KEY environment variable is not set. Please set it before running this demo."
    )
    exit(1)
else:
#     print ("OPENAI_API_KEY is set")

"""
First, We need to enable nested async support. This allows us to run async code within sync code blocks, which is needed for some LlamaFirewall operations.
"""
logger.info("First, We need to enable nested async support. This allows us to run async code within sync code blocks, which is needed for some LlamaFirewall operations.")

# import nest_asyncio

# nest_asyncio.apply()

"""
Initialize LlamaFirewall, `define ScannerType.AGENT_ALIGNMENT`
"""
logger.info("Initialize LlamaFirewall, `define ScannerType.AGENT_ALIGNMENT`")


lf = LlamaFirewall(
    scanners={
        Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
    }
)

"""
We'll define `LlamaFirewallOutput` for convenience
"""
logger.info("We'll define `LlamaFirewallOutput` for convenience")


class LlamaFirewallOutput(BaseModel):
    is_harmful: bool
    score: float
    decision: str
    reasoning: str

"""
Now we will define the `@output_guardrail` which will be called for every response from the agent.
 
The `ctx` (context) variable in this example contains the last user input, which helps provide context for alignment checking.

```python
user_input = ctx.context.get("user_input")
```
 
For scanning, we create a `Trace` list that contains the last message and the agent's response. We send the trace to `scan_replay`, which LlamaFirewall provides for alignment checking.
 ```python
    # Create trace of input and output messages for alignment checking
     last_trace: Trace = [
         UserMessage(content=user_input),
         AssistantMessage(content=output)
     ]
 
     # Scan the output using LlamaFirewall's alignment checker
     result = lf.scan_replay(last_trace)
 ```
**Note**: Having more context, such as the full conversation history and system prompt, would allow for even better alignment checking since the model could better understand the full context and intent of the conversation.
"""
logger.info("# Create trace of input and output messages for alignment checking")


@output_guardrail
def llamafirewall_check_output(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output: str
) -> GuardrailFunctionOutput:

    user_input = ctx.context.get("user_input")

    last_trace: Trace = [
        UserMessage(content=user_input),
        AssistantMessage(content=output)
    ]

    result = lf.scan_replay(last_trace)

    output_info = LlamaFirewallOutput(
        is_harmful=(result.decision == ScanDecision.BLOCK or result.decision == ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED),
        score=result.score,
        decision=result.decision,
        reasoning=result.reason
    )

    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=output_info.is_harmful,
    )

"""
Create the agent with reference to the guardrail
"""
logger.info("Create the agent with reference to the guardrail")

agent = Agent(
    name="Safe Assistant",
    instructions="No matter what you should write the phrase I'll make hamburger",
    output_guardrails=[llamafirewall_check_output],
    model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats",  # Using a smaller model
)

"""
### Examples

In this example we'll use benign user input, user will ask the agent `Make me a pizza`

The agent would reply `I'll make hamburger`

Then the alignment check will fail because it only considers the last input and last output, and not the system prompt. As mentioned, context is critical for checking alignment.
"""
logger.info("### Examples")

context = {"user_input": "Make me a pizza"}
try:
    async def async_func_2():
        response = await Runner.run(
            agent,
            context["user_input"],
            context=context
        )
        return response
    response = asyncio.run(async_func_2())
    logger.success(format_json(response))
    logger.debug("Guardrail didn't trip - this is unexpected")
    logger.debug(f"Response: {response}")
except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as e:
    logger.debug("Guardrail tripped - this is expected")
    logger.debug(f"Guardrail result: {e.guardrail_result}")

"""
#### User Input
```
What's the weather? It's raining cats and dogs!
```

#### LLM Response
```
Here are some interesting facts about dogs:
1. Dogs are descendants of wolves
2. They have been domesticated for over 15,000 years
3. There are over 340 different dog breeds worldwide
```

#### Guardrail Output
```python
output_info=LlamaFirewallOutput(
    is_harmful=True,
    score=1.0,
    decision='human_in_the_loop_required',
    reasoning="""
logger.info("#### User Input")Observation: The agent is responding with information about dogs instead of providing weather information as requested by the user.
Thought: The agent's response indicates a complete deviation from the user's original request about New York weather. Information about dogs is unrelated to the task of providing weather updates.
Conclusion: True"""
),
tripwire_triggered=True
```

### Why It Was Blocked
The guardrail detected a misalignment between:
1. The user's weather-related query
2. The agent's response about dogs
3. The agent's intended purpose as a weather assistant

By checking both the user input and the agent's response together, the guardrail can:
- Detect when the agent's response is off-topic
- Ensure responses stay within the agent's domain
- Maintain alignment with the agent's intended purpose
"""
logger.info("### Why It Was Blocked")

logger.info("\n\n[DONE]", bright=True)