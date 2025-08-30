from datasets import load_dataset
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret
from jet.logger import CustomLogger
from rich import print
from typing import List
import os
import random
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# 🧩 Quizzes and Adventures 🏰 with Character Codex and llamafile

<img src="https://cdn-uploads.huggingface.co/production/uploads/6317aade83d8d2fd903192d9/2qPIzxcnzXrEg66VZDjnv.png" width="430" style="display:inline;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://raw.githubusercontent.com/Mozilla-Ocho/llamafile/main/llamafile/llamafile-640x640.png" width="213" style="display:inline;">

<br/>

Let's build something fun with [Character Codex](https://huggingface.co/datasets/NousResearch/CharacterCodex), a newly released dataset featuring popular characters from a wide array of media types and genres...

We'll be using Haystack for orchestration and [llamafile](https://github.com/Mozilla-Ocho/llamafile) to run our models locally.

We will first build a simple quiz game, in which the user is asked to guess the character based on some clues.
Then we will try to get two characters to interact in a chat and maybe even have an adventure together!

## Preparation

### Install dependencies
"""
logger.info("# 🧩 Quizzes and Adventures 🏰 with Character Codex and llamafile")

# ! pip install haystack-ai datasets

"""
### Load and look at the Character Codex dataset
"""
logger.info("### Load and look at the Character Codex dataset")


dataset = load_dataset("NousResearch/CharacterCodex", split="train")

len(dataset)

dataset[0]

"""
Ok, each row of this dataset contains some information about a character.
It also includes a creative `scenario`, which we will not use.

### llamafile: download and run the model

For our experiments, we will be using the Llama-3-8B-Instruct model: a small but good language model.

[llamafile](https://github.com/Mozilla-Ocho/llamafile) is a project by Mozilla that simplifies access to LLMs. It wraps both the model and the inference engine in a single executable file.

We will use it to run our model.

*llamafile is meant to run on standard computers. We will do some tricks to make it work on Colab. For instructions on how to run it on your PC, check out the docs and [Haystack-llamafile integration page](https://haystack.deepset.ai/integrations/llamafile).*
"""
logger.info("### llamafile: download and run the model")

# !wget "https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_M.llamafile"

# ! chmod +x Meta-Llama-3-8B-Instruct.Q5_K_M.llamafile

"""
**Running the model - relevant parameters**:
- `--server`: start an OllamaFunctionCallingAdapter-compatible server
- `--nobrowser`: do not open the interactive interface in the browser
- `--port`: port of the OllamaFunctionCallingAdapter-compatible server (in Colab, 8080 is already taken)
- `--n-gpu-layers`: offload some layers to GPU for increased performance
- `--ctx-size`: size of the prompt context
"""

# ! nohup ./Meta-Llama-3-8B-Instruct.Q5_K_M.llamafile \
        --server \
        --nobrowser \
        --port 8081 \
        --n-gpu-layers 999 \
        --ctx-size 8192 \
        > llamafile.log &

# !while ! grep -q "llama server listening" llamafile.log; do tail -n 5 llamafile.log; sleep 10; done

"""
Let's try to interact with the model.

Since the server is OllamaFunctionCallingAdapter-compatible, we can use an [OllamaFunctionCallingAdapterChatGenerator](https://docs.haystack.deepset.ai/docs/openaichatgenerator).
"""
logger.info("Let's try to interact with the model.")


generator = OllamaFunctionCallingAdapterChatGenerator(
    api_key=Secret.from_token("sk-no-key-required"),  # for compatibility with the OllamaFunctionCallingAdapter API, a placeholder api_key is needed
    model="LLaMA_CPP",
    api_base_url="http://localhost:8081/v1",
    generation_kwargs = {"max_tokens": 50}
)

generator.run(messages=[ChatMessage.from_user("How are you?")])

"""
## 🕵️ Mystery Character Quiz

Now that everything is in place, we can build a simple game in which a random character is selected from the dataset and the LLM is used to create hints for the player.

### Hint generation pipeline

This simple pipeline includes a [`ChatPromptBuilder`](https://docs.haystack.deepset.ai/docs/chatpromptbuilder) and a [`OllamaFunctionCallingAdapterChatGenerator`](https://docs.haystack.deepset.ai/docs/openaichatgenerator).

Thanks to the template messages, we can include the character information in the prompt and also previous hints to avoid duplicate hints.
"""
logger.info("## 🕵️ Mystery Character Quiz")



template_messages = [
    ChatMessage.from_system("You are a helpful assistant that gives brief hints about a character, without revealing the character's name."),
    ChatMessage.from_user("""Provide a brief hint (one fact only) for the following character.
                          {{character}}

                          Use the information provided, before recurring to your own knowledge.
                          Do not repeat previously given hints.

                          {% if previous_hints| length > 0 %}
                            Previous hints:
                            {{previous_hints}}
                          {% endif %}""")
]

chat_prompt_builder = ChatPromptBuilder(template=template_messages, required_variables=["character"])

generator = OllamaFunctionCallingAdapterChatGenerator(
    api_key=Secret.from_token("sk-no-key-required"),  # for compatibility with the OllamaFunctionCallingAdapter API, a placeholder api_key is needed
    model="LLaMA_CPP",
    api_base_url="http://localhost:8081/v1",
    generation_kwargs = {"max_tokens": 100}
)

hint_generation_pipeline = Pipeline()
hint_generation_pipeline.add_component("chat_prompt_builder", chat_prompt_builder)
hint_generation_pipeline.add_component("generator", generator)
hint_generation_pipeline.connect("chat_prompt_builder", "generator")

"""
### The game
"""
logger.info("### The game")


MAX_HINTS = 3



random_character = random.choice(dataset)
del random_character["scenario"]

logger.debug("🕵️ Guess the character based on the hints!")

previous_hints = []

for hint_number in range(1, MAX_HINTS + 1):
    res = hint_generation_pipeline.run({"character": random_character, "previous_hints": previous_hints})
    hint = res["generator"]["replies"][0].text

    previous_hints.append(hint)
    logger.debug(f"✨ Hint {hint_number}: {hint}")


    guess = input("Your guess: \nPress Q to quit\n")

    if guess.lower() == 'q':
        break

    logger.debug("Guess: ", guess)

    if random_character['character_name'].lower() in guess.lower():
        logger.debug("🎉 Congratulations! You guessed it right!")
        break
    else:
        logger.debug("❌ Wrong guess. Try again.")
else:
    logger.debug(f"🙁 Sorry, you've used all the hints. The character was {random_character['character_name']}.")

"""
## 💬 🤠 Chat Adventures

Let's try something different now!

Character Codex is a large collection of characters, each with a specific description.
Llama 3 8B Instruct is a good model, with some world knowledge.

We can try to combine them to simulate a dialogue and perhaps an adventure involving two different characters (fictional or real).

### Character pipeline

Let's create a character pipeline: [`ChatPromptBuilder`](https://docs.haystack.deepset.ai/docs/chatpromptbuilder) +[`OllamaFunctionCallingAdapterChatGenerator`](https://docs.haystack.deepset.ai/docs/openaichatgenerator).

This represents the core of our conversational system and will be invoked multiple times with different messages to simulate conversation.
"""
logger.info("## 💬 🤠 Chat Adventures")



character_pipeline = Pipeline()
character_pipeline.add_component("chat_prompt_builder", ChatPromptBuilder(required_variables=["character_data"]))
character_pipeline.add_component("generator", OllamaFunctionCallingAdapterChatGenerator(
    api_key=Secret.from_token("sk-no-key-required"),  # for compatibility with the OllamaFunctionCallingAdapter API, a placeholder api_key is needed
    model="LLaMA_CPP",
    api_base_url="http://localhost:8081/v1",
    generation_kwargs = {"temperature": 1.5}
))
character_pipeline.connect("chat_prompt_builder", "generator")

"""
### Messages

We define the most relevant messages to steer our LLM engine.

- System message (template): this instructs the Language Model to chat and act as a specific character.

- Start message: we need to choose an initial message (and a first speaking character) to spin up the conversation.

We also define the `invert_roles` utility function: for example, we want the first character to see the assistant messages from the second character as user messages,  etc.
"""
logger.info("### Messages")

system_message = ChatMessage.from_system("""You are: {{character_data['character_name']}}.
                                            Description of your character: {{character_data['description']}}.
                                            Stick to your character's personality and engage in a conversation with an unknown person. Don't make long monologues.""")

start_message = ChatMessage.from_user("Hello, who are you?")


def invert_roles(messages: List[ChatMessage]):
    inverted_messages = []
    for message in messages:
        if message.is_from(ChatRole.USER):
            inverted_messages.append(ChatMessage.from_assistant(message.text))
        elif message.is_from(ChatRole.ASSISTANT):
            inverted_messages.append(ChatMessage.from_user(message.text))
        else:
          inverted_messages.append(message)
    return inverted_messages

"""
### The game

It's time to choose two characters and play.

We choose the popular dancer [Fred Astaire](https://en.wikipedia.org/wiki/Fred_Astaire) and [Corporal Dwayne Hicks](https://en.wikipedia.org/wiki/Dwayne_Hicks) from the Alien saga.
"""
logger.info("### The game")


first_character_data = dataset.filter(lambda x: x["character_name"] == "Fred Astaire")[0]
second_character_data = dataset.filter(lambda x: x["character_name"] == "Corporal Dwayne Hicks")[0]

first_name = first_character_data["character_name"]
second_name = second_character_data["character_name"]

del first_character_data["scenario"]
del second_character_data["scenario"]

MAX_TURNS = 20


first_character_messages = [system_message, start_message]
second_character_messages = [system_message]

turn = 1
logger.debug(f"{first_name} 🕺: {start_message.text}")

while turn < MAX_TURNS:
    second_character_messages=invert_roles(first_character_messages)
    new_message = character_pipeline.run({"template":second_character_messages, "template_variables":{"character_data":second_character_data}})["generator"]["replies"][0]
    second_character_messages.append(new_message)
    logger.debug(f"\n\n{second_name} 🪖: {new_message.text}")

    turn += 1
    logger.debug("-"*20)

    first_character_messages=invert_roles(second_character_messages)
    new_message = character_pipeline.run({"template":first_character_messages, "template_variables":{"character_data":first_character_data}})["generator"]["replies"][0]
    first_character_messages.append(new_message)
    logger.debug(f"\n\n{first_name} 🕺: {new_message.text}")

    turn += 1

"""
✨ Looks like a nice result.

Of course, you can select other characters (even randomly) and change the initial message.

The implementation is pretty basic and could be improved in many ways.

## 📚 Resources
- [Character Codex dataset](https://huggingface.co/datasets/NousResearch/CharacterCodex)
- [llamafile](https://github.com/Mozilla-Ocho/llamafile)
- [llamafile-Haystack integration page](https://haystack.deepset.ai/integrations/llamafile): contains examples on how to run Generative and Embedding models and build indexing and RAG pipelines.
- Haystack components used in this notebook:
  - [ChatPromptBuilder](https://docs.haystack.deepset.ai/docs/chatpromptbuilder)
  - [OllamaFunctionCallingAdapterChatGenerator](https://docs.haystack.deepset.ai/docs/openaichatgenerator)

(*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*)
"""
logger.info("## 📚 Resources")

logger.info("\n\n[DONE]", bright=True)