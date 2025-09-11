from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
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
---
sidebar_position: 2
---

# How to use few shot examples in chat models

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Prompt templates](/docs/concepts/prompt_templates)
- [Example selectors](/docs/concepts/example_selectors)
- [Chat models](/docs/concepts/chat_models)
- [Vectorstores](/docs/concepts/vectorstores)

:::

This guide covers how to prompt a chat model with example inputs and outputs. Providing the model with a few such examples is called [few-shotting](/docs/concepts/few_shot_prompting/), and is a simple yet powerful way to guide generation and in some cases drastically improve model performance.

There does not appear to be solid consensus on how best to do few-shot prompting, and the optimal prompt compilation will likely vary by model. Because of this, we provide few-shot prompt templates like the [FewShotChatMessagePromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate.html?highlight=fewshot#langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate) as a flexible starting point, and you can modify or replace them as you see fit.

The goal of few-shot prompt templates are to dynamically select examples based on an input, and then format the examples in a final prompt to provide for the model.

**Note:** The following code examples are for chat models only, since `FewShotChatMessagePromptTemplates` are designed to output formatted [chat messages](/docs/concepts/messages) rather than pure strings. For similar few-shot prompt examples for pure string templates compatible with completion models (LLMs), see the [few-shot prompt templates](/docs/how_to/few_shot_examples/) guide.

## Fixed Examples

The most basic (and common) few-shot prompting technique is to use fixed prompt examples. This way you can select a chain, evaluate it, and avoid worrying about additional moving parts in production.

The basic components of the template are:
- `examples`: A list of dictionary examples to include in the final prompt.
- `example_prompt`: converts each example into 1 or more messages through its [`format_messages`](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html?highlight=format_messages#langchain_core.prompts.chat.ChatPromptTemplate.format_messages) method. A common example would be to convert each example into one human message and one AI message response, or a human message followed by a function call message.

Below is a simple demonstration. First, define the examples you'd like to include. Let's give the LLM an unfamiliar mathematical operator, denoted by the "🦜" emoji:
"""
logger.info("# How to use few shot examples in chat models")

# %pip install -qU langchain langchain-ollama langchain-chroma

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

"""
If we try to ask the model what the result of this expression is, it will fail:
"""
logger.info("If we try to ask the model what the result of this expression is, it will fail:")


model = ChatOllama(model="llama3.2")

model.invoke("What is 2 🦜 9?")

"""
Now let's see what happens if we give the LLM some examples to work with. We'll define some below:
"""
logger.info("Now let's see what happens if we give the LLM some examples to work with. We'll define some below:")


examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
]

"""
Next, assemble them into the few-shot prompt template.
"""
logger.info("Next, assemble them into the few-shot prompt template.")

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

logger.debug(few_shot_prompt.invoke({}).to_messages())

"""
Finally, we assemble the final prompt as shown below, passing `few_shot_prompt` directly into the `from_messages` factory method, and use it with a model:
"""
logger.info("Finally, we assemble the final prompt as shown below, passing `few_shot_prompt` directly into the `from_messages` factory method, and use it with a model:")

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

"""
And now let's ask the model the initial question and see how it does:
"""
logger.info("And now let's ask the model the initial question and see how it does:")


chain = final_prompt | model

chain.invoke({"input": "What is 2 🦜 9?"})

"""
And we can see that the model has now inferred that the parrot emoji means addition from the given few-shot examples!

## Dynamic few-shot prompting

Sometimes you may want to select only a few examples from your overall set to show based on the input. For this, you can replace the `examples` passed into `FewShotChatMessagePromptTemplate` with an `example_selector`. The other components remain the same as above! Our dynamic few-shot prompt template would look like:

- `example_selector`: responsible for selecting few-shot examples (and the order in which they are returned) for a given input. These implement the [BaseExampleSelector](https://python.langchain.com/api_reference/core/example_selectors/langchain_core.example_selectors.base.BaseExampleSelector.html?highlight=baseexampleselector#langchain_core.example_selectors.base.BaseExampleSelector) interface. A common example is the vectorstore-backed [SemanticSimilarityExampleSelector](https://python.langchain.com/api_reference/core/example_selectors/langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector.html?highlight=semanticsimilarityexampleselector#langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector)
- `example_prompt`: convert each example into 1 or more messages through its [`format_messages`](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html?highlight=chatprompttemplate#langchain_core.prompts.chat.ChatPromptTemplate.format_messages) method. A common example would be to convert each example into one human message and one AI message response, or a human message followed by a function call message.

These once again can be composed with other messages and chat templates to assemble your final prompt.

Let's walk through an example with the `SemanticSimilarityExampleSelector`. Since this implementation uses a vectorstore to select examples based on semantic similarity, we will want to first populate the store. Since the basic idea here is that we want to search for and return examples most similar to the text input, we embed the `values` of our prompt examples rather than considering the keys:
"""
logger.info("## Dynamic few-shot prompting")


examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
    {"input": "2 🦜 4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

"""
### Create the `example_selector`

With a vectorstore created, we can create the `example_selector`. Here we will call it in isolation, and set `k` on it to only fetch the two example closest to the input.
"""
logger.info("### Create the `example_selector`")

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

example_selector.select_examples({"input": "horse"})

"""
### Create prompt template

We now assemble the prompt template, using the `example_selector` created above.
"""
logger.info("### Create prompt template")


few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

logger.debug(few_shot_prompt.invoke(input="What's 3 🦜 3?").to_messages())

"""
And we can pass this few-shot chat message prompt template into another chat prompt template:
"""
logger.info("And we can pass this few-shot chat message prompt template into another chat prompt template:")

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

logger.debug(few_shot_prompt.invoke(input="What's 3 🦜 3?"))

"""
### Use with an chat model

Finally, you can connect your model to the few-shot prompt.
"""
logger.info("### Use with an chat model")

chain = final_prompt | ChatOllama(model="llama3.2")

chain.invoke({"input": "What's 3 🦜 3?"})

"""
## Next steps

You've now learned how to add few-shot examples to your chat prompts.

Next, check out the other how-to guides on prompt templates in this section, the related how-to guide on [few shotting with text completion models](/docs/how_to/few_shot_examples), or the other [example selector how-to guides](/docs/how_to/example_selectors/).
"""
logger.info("## Next steps")


logger.info("\n\n[DONE]", bright=True)