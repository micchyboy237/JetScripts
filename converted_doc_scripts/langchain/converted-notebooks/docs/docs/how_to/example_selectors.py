from jet.logger import logger
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
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
sidebar_position: 1
---

# How to use example selectors

If you have a large number of examples, you may need to select which ones to include in the prompt. The [Example Selector](/docs/concepts/example_selectors/) is the class responsible for doing so.

The base interface is defined as below:

```python
class BaseExampleSelector(ABC):
    """
logger.info("# How to use example selectors")Interface for selecting examples to include in prompts."""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """
logger.info("def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:")Select which examples to use based on the inputs."""
        
    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> Any:
        """
logger.info("def add_example(self, example: Dict[str, str]) -> Any:")Add new example to store."""
```

The only method it needs to define is a ``select_examples`` method. This takes in the input variables and then returns a list of examples. It is up to each specific implementation as to how those examples are selected.

LangChain has a few different types of example selectors. For an overview of all these types, see the [below table](#example-selector-types).

In this guide, we will walk through creating a custom example selector.

## Examples

In order to use an example selector, we need to create a list of examples. These should generally be example inputs and outputs. For this demo purpose, let's imagine we are selecting examples of how to translate English to Italian.
"""
logger.info("## Examples")

examples = [
    {"input": "hi", "output": "ciao"},
    {"input": "bye", "output": "arrivederci"},
    {"input": "soccer", "output": "calcio"},
]

"""
## Custom Example Selector

Let's write an example selector that chooses what example to pick based on the length of the word.
"""
logger.info("## Custom Example Selector")



class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        new_word = input_variables["input"]
        new_word_length = len(new_word)

        best_match = None
        smallest_diff = float("inf")

        for example in self.examples:
            current_diff = abs(len(example["input"]) - new_word_length)

            if current_diff < smallest_diff:
                smallest_diff = current_diff
                best_match = example

        return [best_match]

example_selector = CustomExampleSelector(examples)

example_selector.select_examples({"input": "okay"})

example_selector.add_example({"input": "hand", "output": "mano"})

example_selector.select_examples({"input": "okay"})

"""
## Use in a Prompt

We can now use this example selector in a prompt
"""
logger.info("## Use in a Prompt")


example_prompt = PromptTemplate.from_template("Input: {input} -> Output: {output}")

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Input: {input} -> Output:",
    prefix="Translate the following words from English to Italian:",
    input_variables=["input"],
)

logger.debug(prompt.format(input="word"))

"""
## Example Selector Types

| Name       | Description                                                                                 |
|------------|---------------------------------------------------------------------------------------------|
| Similarity | Uses semantic similarity between inputs and examples to decide which examples to choose.    |
| MMR        | Uses Max Marginal Relevance between inputs and examples to decide which examples to choose. |
| Length     | Selects examples based on how many can fit within a certain length                          |
| Ngram      | Uses ngram overlap between inputs and examples to decide which examples to choose.          |
"""
logger.info("## Example Selector Types")


logger.info("\n\n[DONE]", bright=True)