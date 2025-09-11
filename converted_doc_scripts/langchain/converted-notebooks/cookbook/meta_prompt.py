from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Meta-Prompt

This is a LangChain implementation of [Meta-Prompt](https://noahgoodman.substack.com/p/meta-prompt-a-simple-self-improving), by [Noah Goodman](https://cocolab.stanford.edu/ndg), for building self-improving agents.

The key idea behind Meta-Prompt is to prompt the agent to reflect on its own performance and modify its own instructions.

![figure](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F468217b9-96d9-47c0-a08b-dbf6b21b9f49_492x384.png)

Here is a description from the [original blog post](https://noahgoodman.substack.com/p/meta-prompt-a-simple-self-improving):


The agent is a simple loop that starts with no instructions and follows these steps:

Engage in conversation with a user, who may provide requests, instructions, or feedback.

At the end of the episode, generate self-criticism and a new instruction using the meta-prompt
```
Assistant has just had the below interactions with a User. Assistant followed their "system: Instructions" closely. Your job is to critique the Assistant's performance and then revise the Instructions so that Assistant would quickly and correctly respond in the future.
 
####
{hist}
####
 
Please reflect on these interactions.

You should first critique Assistant's performance. What could Assistant have done better? What should the Assistant remember about this user? Are there things this user always wants? Indicate this with "Critique: ...".

You should next revise the Instructions so that Assistant would quickly and correctly respond in the future. Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions. Don't forget any important details in the current Instructions! Indicate the new Instructions by "Instructions: ...".
```

Repeat.

The only fixed instructions for this system (which I call Meta-prompt) is the meta-prompt that governs revision of the agent’s instructions. The agent has no memory between episodes except for the instruction it modifies for itself each time. Despite its simplicity, this agent can learn over time and self-improve by incorporating useful details into its instructions.

## Setup
We define two chains. One serves as the `Assistant`, and the other is a "meta-chain" that critiques the `Assistant`'s performance and modifies the instructions to the `Assistant`.
"""
logger.info("# Meta-Prompt")


def initialize_chain(instructions, memory=None):
    if memory is None:
        memory = ConversationBufferWindowMemory()
        memory.ai_prefix = "Assistant"

    template = f"""
    Instructions: {instructions}
    {{{memory.memory_key}}}
    Human: {{human_input}}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["history", "human_input"], template=template
    )

    chain = LLMChain(
        llm=Ollama(temperature=0),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(),
    )
    return chain


def initialize_meta_chain():
    meta_template = """
    Assistant has just had the below interactions with a User. Assistant followed their "Instructions" closely. Your job is to critique the Assistant's performance and then revise the Instructions so that Assistant would quickly and correctly respond in the future.


    {chat_history}


    Please reflect on these interactions.

    You should first critique Assistant's performance. What could Assistant have done better? What should the Assistant remember about this user? Are there things this user always wants? Indicate this with "Critique: ...".

    You should next revise the Instructions so that Assistant would quickly and correctly respond in the future. Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions. Don't forget any important details in the current Instructions! Indicate the new Instructions by "Instructions: ...".
    """

    meta_prompt = PromptTemplate(
        input_variables=["chat_history"], template=meta_template
    )

    meta_chain = LLMChain(
        llm=Ollama(temperature=0),
        prompt=meta_prompt,
        verbose=True,
    )
    return meta_chain


def get_chat_history(chain_memory):
    memory_key = chain_memory.memory_key
    chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
    return chat_history


def get_new_instructions(meta_output):
    delimiter = "Instructions: "
    new_instructions = meta_output[meta_output.find(
        delimiter) + len(delimiter):]
    return new_instructions


def main(task, max_iters=3, max_meta_iters=5):
    failed_phrase = "task failed"
    success_phrase = "task succeeded"
    key_phrases = [success_phrase, failed_phrase]

    instructions = "None"
    for i in range(max_meta_iters):
        logger.debug(f"[Episode {i + 1}/{max_meta_iters}]")
        chain = initialize_chain(instructions, memory=None)
        output = chain.predict(human_input=task)
        for j in range(max_iters):
            logger.debug(f"(Step {j + 1}/{max_iters})")
            logger.debug(f"Assistant: {output}")
            logger.debug("Human: ")
            human_input = input()
            if any(phrase in human_input.lower() for phrase in key_phrases):
                break
            output = chain.predict(human_input=human_input)
        if success_phrase in human_input.lower():
            logger.debug("You succeeded! Thanks for playing!")
            return
        meta_chain = initialize_meta_chain()
        meta_output = meta_chain.predict(
            chat_history=get_chat_history(chain.memory))
        logger.debug(f"Feedback: {meta_output}")
        instructions = get_new_instructions(meta_output)
        logger.debug(f"New Instructions: {instructions}")
        logger.debug("\n" + "#" * 80 + "\n")
    logger.debug("You failed! Thanks for playing!")


"""
## Specify a task and interact with the agent
"""
logger.info("## Specify a task and interact with the agent")

task = "Provide a systematic argument for why we should always eat pasta with olives."
main(task)

logger.info("\n\n[DONE]", bright=True)
