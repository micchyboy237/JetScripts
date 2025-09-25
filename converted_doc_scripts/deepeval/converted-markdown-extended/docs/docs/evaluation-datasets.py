from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.dataset import EvaluationDataset, ConversationalGolden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ConversationalRelevancyMetric
from deepeval.simulator import ConversationSimulator
from deepeval.synthesizer import Synthesizer
from deepeval.test_case import ConversationalTestCase
from deepeval.test_case import LLMTestCase
from jet.logger import logger
from pydantic import BaseModel
import TabItem from "@theme/TabItem";
import Tabs from "@theme/Tabs";
import VideoDisplayer from "@site/src/components/VideoDisplayer";
import os
import pytest
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
id: evaluation-datasets
title: Datasets
sidebar_label: Datasets
---


<head>
  <link rel="canonical" href="https://deepeval.com/docs/evaluation-datasets" />
</head>

In `deepeval`, an evaluation dataset, or just dataset, is a collection of goldens. A golden is a precursor to a test case. At evaluation time, you would first convert all goldens in your dataset to test cases, before running evals on these test cases.

## Quick Summary

There are two approaches to running evals using datasets in `deepeval`:

1. Using `deepeval test run`
2. Using `evaluate`

Depending on the type of goldens you supply, datasets are either **single-turn** or **mult-turn**. Evaluating a dataset means exactly the same as evaluating your LLM system, because by definition a dataset contains all the information produced by your LLM needed for evaluation.

<details>

<summary>
  What are the best practices for curating an evaluation dataset?
</summary>

- **Ensure telling test coverage:** Include diverse real-world inputs, varying complexity levels, and edge cases to properly challenge the LLM.
- **Focused, quantitative test cases:** Design with clear scope that enables meaningful performance metrics without being too broad or narrow.
- **Define clear objectives:** Align datasets with specific evaluation goals while avoiding unnecessary fragmentation.

</details>

:::info

If you don't already have an `EvaluationDataset`, a great starting point is to simply write down the prompts you're currently using to manually eyeball your LLM outputs. You can also do this on Confident AI, which integrates 100% with `deepeval`:

<VideoDisplayer
  src="https://confident-docs.s3.us-east-1.amazonaws.com/dataset-editor:dataset-annotation.mp4"
  confidentUrl="/docs/dataset-editor/annotate-datasets"
  label="Learn Dataset Annotation on Confident AI"
/>

Full documentation for datasets on [Confident AI
here.](https://www.confident-ai.com/docs/llm-evaluation/dataset-management/create-goldens)

:::

## Create A Dataset

An `EvaluationDataset` in `deepeval` is simply a collection of goldens. You can initialize an empty dataset to start with:
"""
logger.info("## Quick Summary")


dataset = EvaluationDataset()

"""
A dataset can either be a single-turn one, **or** a multi-turn one (but not both). During initialization supplying your dataset with a list of `Golden`s will make it a single-turn one, whereas supplying it with `ConversationalGolden`s will make it multi-turn:

<Tabs groupId="single-multi-turns">

<TabItem value="single-turn" label="Single-Turn">
"""
logger.info("A dataset can either be a single-turn one, **or** a multi-turn one (but not both). During initialization supplying your dataset with a list of `Golden`s will make it a single-turn one, whereas supplying it with `ConversationalGolden`s will make it multi-turn:")


dataset = EvaluationDataset(goldens=[Golden(input="What is your name?")])
logger.debug(dataset._multi_turn) # prints False

"""
</TabItem>

<TabItem value="multi-turn" label="Multi-Turn">
"""


dataset = EvaluationDataset(
    goldens=[
        ConversationalGolden(
            scenario="Frustrated user asking for a refund.",
            expected_outcome="Redirected to a human agent."
        )
    ]
)
logger.debug(dataset._multi_turn) # prints True

"""
</TabItem>

</Tabs>

To ensure best practices, datasets in `deepeval` are stateful and opinionated. This means you cannot change the value of `_multi_turn` once its value has been set. However, you can always add new goldens after initialization using the `add_golden` method:

<Tabs groupId="single-multi-turns">

<TabItem value="single-turn" label="Single-Turn">
"""
logger.info("To ensure best practices, datasets in `deepeval` are stateful and opinionated. This means you cannot change the value of `_multi_turn` once its value has been set. However, you can always add new goldens after initialization using the `add_golden` method:")

...

dataset.add_golden(Golden(input="Nice."))

"""
</TabItem>

<TabItem value="multi-turn" label="Multi-Turn">
"""

...

dataset.add_golden(
    ConversationalGolden(
        scenario="User expressing gratitude for redirecting to human.",
        expected_outcome="Appreciates the gratitude."
    )
)

"""
</TabItem>

</Tabs>

## Run Evals On Dataset

You run evals on test cases in datasets, which you'll create at evaluation time using the goldens in the same dataset.

![Evaluation Dataset](https://deepeval-docs.s3.us-east-1.amazonaws.com/docs:evaluation-dataset.png)

First step is to load in the goldens to your dataset. This example will load datasets from Confident AI, but you can also explore [other options below.](#load-dataset)
"""
logger.info("## Run Evals On Dataset")


dataset = EvaluationDataset()
dataset.pull(alias="My Dataset") # replace with your alias
logger.debug(dataset.goldens) # print to sanity check yourself

"""
:::tip
Your dataset is either single or multi-turn the moment you pull your dataset.
:::

Once you have your dataset and can see a non-empty list of goldens, you can start generating outputs and **add it back to your dataset** as test cases via the `add_test_case()` method:

<Tabs groupId="single-multi-turns">

<TabItem value="single-turn" label="Single-Turn">
"""
logger.info("Your dataset is either single or multi-turn the moment you pull your dataset.")

...

for golden in dataset.goldens:
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=your_llm_app(golden.input) # replace with your LLM app
    )
    dataset.add_test_case(test_case)

logger.debug(dataset.test_cases) # print to santiy check yourself

"""
Lastly, you can run evaluations on the list of test cases in your dataset:

<Tabs>

<TabItem value="ci-cd" label="Unit-Testing In CI/CD">
"""
logger.info("Lastly, you can run evaluations on the list of test cases in your dataset:")

...

@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_llm_app(test_case: LLMTestCase):
    assert_test(test_case=test_case, metrics=[AnswerRelevancyMetric()])

"""
And execute the test file:
"""
logger.info("And execute the test file:")

deepeval test run test_llm_app.py

"""
You can learn more about `assert_test` in [this section.](/docs/evaluation-end-to-end-llm-evals#use-deepeval-test-run-in-cicd-pipelines)

</TabItem>

<TabItem value="python" label="In Python Scripts">
"""
logger.info("You can learn more about `assert_test` in [this section.](/docs/evaluation-end-to-end-llm-evals#use-deepeval-test-run-in-cicd-pipelines)")

...

evaluate(test_cases=dataset.test_cases, metrics=[AnswerRelevancyMetric()])

"""
And run `main.py`:
"""
logger.info("And run `main.py`:")

python main.py

"""
You can learn more about `evaluate` in [this section.](/docs/evaluation-end-to-end-llm-evals#use-evaluate-in-python-scripts)

</TabItem>

</Tabs>

</TabItem>

<TabItem value="multi-turn" label="Multi-Turn">
"""
logger.info("You can learn more about `evaluate` in [this section.](/docs/evaluation-end-to-end-llm-evals#use-evaluate-in-python-scripts)")

...

for golden in dataset.goldens:
    test_case = ConversationalTestCase(
        scenario=golden.scenario,
        turns=generate_turns(golden.scenario) # replace with your method to simulate conversations
    )
    dataset.add_test_case(test_case)

logger.debug(dataset.test_cases) # print to santiy check yourself

"""
Lastly, you can run evaluations on the list of test cases in your dataset:

<Tabs>

<TabItem value="ci-cd" label="Unit-Testing In CI/CD">
"""
logger.info("Lastly, you can run evaluations on the list of test cases in your dataset:")

...

@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_llm_app(test_case: ConversationalTestCase):
    assert_test(test_case=test_case, metrics=[ConversationalRelevancyMetric()])

"""
And execute the test file:
"""
logger.info("And execute the test file:")

deepeval test run test_llm_app.py

"""
You can learn more about `assert_test` in [this section.](/docs/evaluation-end-to-end-llm-evals#use-deepeval-test-run-in-cicd-pipelines)

</TabItem>

<TabItem value="python" label="In Python Scripts">
"""
logger.info("You can learn more about `assert_test` in [this section.](/docs/evaluation-end-to-end-llm-evals#use-deepeval-test-run-in-cicd-pipelines)")

...

evaluate(test_cases=dataset.test_cases, metrics=[ConversationalRelevancyMetric()])

"""
And run `main.py`:
"""
logger.info("And run `main.py`:")

python main.py

"""
You can learn more about `evaluate` in [this section.](/docs/evaluation-end-to-end-llm-evals#use-evaluate-in-python-scripts)

</TabItem>

</Tabs>

</TabItem>

</Tabs>

## Manage Your Dataset

Dataset management is an essential part of your evaluation lifecycle. We recommend Confident AI as the choice for your dataset management workflow as it comes with dozens of collaboration features out of the box, but you can also do it locally as well.

### Save Dataset

You can store both single-turn and multi-turn datasets with `deepeval`. The single-turn datasets contains a list of `Golden`s and the multi-turn would contain `ConversationalGolden`s instead.

<Tabs>

<TabItem value="confident-ai" label="Confident AI">

You can save your dataset on the cloud by using the `push` method:
"""
logger.info("## Manage Your Dataset")


dataset = EvaluationDataset(goldens)
dataset.push(alias="My dataset")

"""
This pushes all goldens in your evaluation dataset to Confident AI. If you don't already have a dataset, the `push` method will automatically create one. If you already have a dataset with the same alias, you can also choose to **optionally** overwrite it by setting `overwrite` to `True`:
"""
logger.info("This pushes all goldens in your evaluation dataset to Confident AI. If you don't already have a dataset, the `push` method will automatically create one. If you already have a dataset with the same alias, you can also choose to **optionally** overwrite it by setting `overwrite` to `True`:")

...
dataset.push(alias="My dataset")

"""
Lastly, if you're unsure whether your goldens are ready for evaluation, you should set `finalized` to `False` instead:
"""
logger.info("Lastly, if you're unsure whether your goldens are ready for evaluation, you should set `finalized` to `False` instead:")

...

dataset.push(alias="My dataset", finalized=False)

"""
The `queue` method will similarly push goldens but will not mark it as "finalized" on Confident AI. This means they won't be pulled until you've manually marked them as finalized on the platform. You can learn more on Confident AI's docs [here.](https://www.confident-ai.com/docs/llm-evaluation/dataset-management/create-goldens)

:::tip
You can also push multi-turn datasets.
:::

</TabItem>

<TabItem value="local-json" label="Locally as JSON">

You can save your dataset locally to a JSON file by using the `save_as()` method:
"""
logger.info("The `queue` method will similarly push goldens but will not mark it as "finalized" on Confident AI. This means they won't be pulled until you've manually marked them as finalized on the platform. You can learn more on Confident AI's docs [here.](https://www.confident-ai.com/docs/llm-evaluation/dataset-management/create-goldens)")


dataset = EvaluationDataset(goldens)
dataset.save_as(
    file_type="json",
    directory="./deepeval-test-dataset",
)

"""
There are **TWO** mandatory and **TWO** optional parameter when calling the `save_as()` method:

- `file_type`: a string of either `"csv"` or `"json"` and specifies which file format to save `Golden`s in.
- `directory`: a string specifying the path of the directory you wish to save `Golden`s at.
- `file_name`: a string specifying the custom filename for the dataset file. Defaulted to the "YYYYMMDD_HHMMSS" format of time now.
- `include_test_cases`: a boolean which when set to `True`, will also save any test cases within your dataset. Defaulted to `False`.

:::note
By default the `save_as()` method only saves the `Golden`s within your `EvaluationDataset` to file. If you wish to save test cases as well, set `include_test_cases` to `True`.
:::

</TabItem>
<TabItem value="local-csv" label="Locally as CSV">

You can save your dataset locally to a CSV file by using the `save_as()` method:
"""
logger.info("There are **TWO** mandatory and **TWO** optional parameter when calling the `save_as()` method:")


dataset = EvaluationDataset(goldens)
dataset.save_as(
    file_type="csv",
    directory="./deepeval-test-dataset",
)

"""
There are **TWO** mandatory and **TWO** optional parameter when calling the `save_as()` method:

- `file_type`: a string of either `"csv"` or `"json"` and specifies which file format to save `Golden`s in.
- `directory`: a string specifying the path of the directory you wish to save `Golden`s at.
- `file_name`: a string specifying the custom filename for the dataset file. Defaulted to the "YYYYMMDD_HHMMSS" format of time now.
- `include_test_cases`: a boolean which when set to `True`, will also save any test cases within your dataset. Defaulted to `False`.

:::note
By default the `save_as()` method only saves the `Golden`s within your `EvaluationDataset` to file. If you wish to save test cases as well, set `include_test_cases` to `True`.
:::

</TabItem>

</Tabs>

### Load Dataset

`deepeval` offers support for loading datasets stored in JSON files, CSV files, and hugging face datasets into an `EvaluationDataset` as either test cases or goldens.

<Tabs>

<TabItem value="confident-ai" label="Confident AI">

You can load entire datasets on Confident AI's cloud in one line of code.
"""
logger.info("### Load Dataset")


dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")

"""
Non-technical domain experts can **create, annotate, and comment** on datasets on Confident AI. You can also upload datasets in CSV format, or push synthetic datasets created in `deepeval` to Confident AI in one line of code.

For more information, visit the [Confident AI datasets section.](https://www.confident-ai.com/docs/llm-evaluation/dataset-management/create-goldens)

</TabItem>

<TabItem value="from-json" label="From JSON">

You can loading an existing `EvaluationDataset` you might have generated elsewhere by supplying a `file_path` to your `.json` file as **either test cases or goldens**. Your `.json` file should contain an array of objects (or list of dictionaries).
"""
logger.info("Non-technical domain experts can **create, annotate, and comment** on datasets on Confident AI. You can also upload datasets in CSV format, or push synthetic datasets created in `deepeval` to Confident AI in one line of code.")


dataset = EvaluationDataset()

dataset.add_goldens_from_json_file(
    file_path="example.json",
) # file_path is the absolute path to your .json file

"""
If your JSON file has different keys from `deepeval`'s conventional `Golden` or `ConversationalGolden` parameters. You can supply your custom key names in the [function parameters](https://github.com/confident-ai/deepeval/blob/main/deepeval/dataset/dataset.py#L584).

You can also add single-turn `LLMTestCase`s to your dataset from a JSON file.
"""
logger.info("If your JSON file has different keys from `deepeval`'s conventional `Golden` or `ConversationalGolden` parameters. You can supply your custom key names in the [function parameters](https://github.com/confident-ai/deepeval/blob/main/deepeval/dataset/dataset.py#L584).")


dataset = EvaluationDataset()

dataset.add_test_cases_from_json_file(
    file_path="example.json",
    input_key_name="query",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context",
    retrieval_context_key_name="retrieval_context",
)

"""
:::info
Loading datasets as goldens are especially helpful if you're looking to generate LLM `actual_output`s at evaluation time. You might find yourself in this situation if you are generating data for testing or using historical data from production.
:::

</TabItem>

<TabItem value="from-csv" label="From CSV">

You can add test cases or goldens into your `EvaluationDataset` by supplying a `file_path` to your `.csv` file. Your `.csv` file should contain rows that can be mapped into `Golden` or `ConversationalGolden` through their column names.

Remember, parameters such as `context` should be a list of strings and in the context of CSV files, it means you have to supply a `context_col_delimiter` argument to tell `deepeval` how to split your context cells into a list of strings.
"""
logger.info("Loading datasets as goldens are especially helpful if you're looking to generate LLM `actual_output`s at evaluation time. You might find yourself in this situation if you are generating data for testing or using historical data from production.")


dataset = EvaluationDataset()

dataset.add_goldens_from_csv_file(
    file_path="example.csv",
) # file_path is the absolute path to you .csv file

"""
If your CSV file has different column names from `deepeval`'s conventional `Golden` or `ConversationalGolden` parameters. You can supply your custom column names in the [function parameters](https://github.com/confident-ai/deepeval/blob/main/deepeval/dataset/dataset.py#L433).

You can also add single-turn `LLMTestCase`s to your dataset from a CSV file.
"""
logger.info("If your CSV file has different column names from `deepeval`'s conventional `Golden` or `ConversationalGolden` parameters. You can supply your custom column names in the [function parameters](https://github.com/confident-ai/deepeval/blob/main/deepeval/dataset/dataset.py#L433).")


dataset = EvaluationDataset()

dataset.add_test_cases_from_csv_file(
    file_path="example.csv",
    input_col_name="query",
    actual_output_col_name="actual_output",
    expected_output_col_name="expected_output",
    context_col_name="context",
    context_col_delimiter= ";",
    retrieval_context_col_name="retrieval_context",
    retrieval_context_col_delimiter= ";"
)

"""
:::note
Since `expected_output`, `context`, `retrieval_context`, `tools_called`, and `expected_tools` are optional parameters for an `LLMTestCase`, these fields are similarly **optional** parameters when adding test cases from an existing dataset.
:::

</TabItem>

</Tabs>

## Generate A Dataset

Sometimes, you might not have datasets ready to use, and that's ok. `deepeval` provides two options for both single-turn and multi-turn use cases:

- `Synthesizer` for generating single-turn goldens
- `ConversationSimulator` for generating `turn`s in a [`ConversationalTestCase`](/docs/evaluation-multiturn-test-cases#conversational-test-case)

### Synthesizer

`deepeval` offers anyone the ability to easily generate synthetic datasets from documents locally on your machine. This is especially helpful if you don't have an evaluation dataset prepared beforehand.
"""
logger.info("## Generate A Dataset")


goldens = Synthesizer().generate_goldens_from_docs(
    document_paths=['example.txt', 'example.docx', 'example.pdf']
)

dataset = EvaluationDataset(goldens=goldens)

"""
In this example, we've used the `generate_goldens_from_docs` method, which is one of the four generation methods offered by `deepeval`'s `Synthesizer`. The four methods include:

- [`generate_goldens_from_docs()`](/docs/synthesizer-generate-from-docs): useful for generating goldens to evaluate your LLM application based on contexts extracted from your knowledge base in the form of documents.
- [`generate_goldens_from_contexts()`](/docs/synthesizer-generate-from-contexts): useful for generating goldens to evaluate your LLM application based on a list of prepared context.
- [`generate_goldens_from_scratch()`](/docs/synthesizer-generate-from-scratch): useful for generating goldens to evaluate your LLM application without relying on contexts from a knowledge base.
- [`generate_goldens_from_goldens()`](/docs/synthesizer-generate-from-goldens): useful for generating goldens by augmenting a known set of goldens.

`deepeval`'s `Synthesizer` uses a series of evolution techniques to complicate and make generated goldens more realistic to human prepared data.

:::info
For more information on how `deepeval`'s `Synthesizer` works, visit the [synthesizer section.](/docs/synthesizer-introduction#how-does-it-work)
:::

### Conversation Simulator

While a `Synthesizer` generates goldens, the `ConversationSimulator` works slightly different as it generates `turns` in a `ConversationalTestCase` instead:
"""
logger.info("### Conversation Simulator")


simulator = ConversationSimulator(
    user_intentions={"Opening a bank account": 1},
    user_profile_items=[
        "full name",
        "current address",
        "bank account number",
        "date of birth",
        "mother's maiden name",
        "phone number",
        "country code",
    ],
)

async def model_callback(input: str, conversation_history: List[Dict[str, str]]) -> str:
    return f"I don't know how to answer this: {input}"

convo_test_cases = simulator.simulate(
  model_callback=model_callback,
  stopping_criteria="Stop when the user's banking request has been fully resolved.",
)
logger.debug(convo_test_cases)

"""
You can learn more in the [conversation simulator page.](/docs/conversation-simulator)

## What Are Goldens?

Goldens represent a more flexible alternative to test cases in the `deepeval`, and **is the preferred way to initialize a dataset**. Unlike test cases, goldens:

- Only require `input`/`scenario` to initialize
- Store expected results like `expected_output`/`expected_outcome`
- Serve as templates before becoming fully-formed test cases

Goldens excel in development workflows where you need to:

- Evaluate changes across different iterations of your LLM application
- Compare performance between model versions
- Test with `input`s that haven't yet been processed by your LLM

Think of goldens as "pending test cases" - they contain all the input data and expected results, but are missing the dynamic elements (`actual_output`, `retrieval_context`, `tools_called`) that will be generated when your LLM processes them.

### Data model

The golden data model is nearly identical to their single/multi-turn test case counterparts (aka. `LLMTestCase` and `ConversationalTestCase`).

For single-turn `Golden`s:
"""
logger.info("## What Are Goldens?")


class Golden(BaseModel):
    input: str
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    expected_tools: Optional[List[ToolCall]] = None

    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    custom_column_key_values: Optional[Dict[str, str]] = None

    actual_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[ToolCall]] = None

"""
:::info
The `actual_output`, `retrieval_context`, and `tools_called` are meant to be populated dynamically instead of passed directly from a golden to test case at evaluation time.
:::

For multi-turn `ConversationalGolden`s:
"""
logger.info("The `actual_output`, `retrieval_context`, and `tools_called` are meant to be populated dynamically instead of passed directly from a golden to test case at evaluation time.")


class ConversationalGolden(BaseModel):
    scenario: str
    expected_outcome: Optional[str] = None
    user_description: Optional[str] = None
    context: Optional[List[str]] = None

    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    custom_column_key_values: Optional[Dict[str, str]] = None

    turns: Optional[Turn] = None

"""
You can easily add and edit custom columns on [Confident AI.](https://www.confident-ai.com/docs/llm-evaluation/dataset-management/create-goldens#custom-dataset-columns)

:::tip

The `turns` parameter should **100%** be generated at evaluation time in your `ConversationalTestCase` instead. However, the `turns` parameter exists in case users want to either:

- [Simulate turns](/docs/conversation-simulator) starting from a certain point of a prior conversation that was previously left off
- Continue from a specific turn when test cases usually fail at the last turn where agents are calling multiple tools

:::
"""
logger.info("You can easily add and edit custom columns on [Confident AI.](https://www.confident-ai.com/docs/llm-evaluation/dataset-management/create-goldens#custom-dataset-columns)")

logger.info("\n\n[DONE]", bright=True)