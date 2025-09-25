from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from jet.logger import logger
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
id: guides-answer-correctness-metric
title: Answer Correctness Metric
sidebar_label: Answer Correctness Metric
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/guides/guides-answer-correctness-metric"
  />
</head>

**Answer Correctness** (or Correctness) is one of the most important and commonly used evaluation metrics for LLM applications. Correctness is typically scored from 0 to 1, with 1 indicating a correct answer and 0 indicating an incorrect one.

:::info
Although numerous general-purpose Correctness metrics exist, our users find it most useful to create a **custom Correctness metric** for their custom LLM application. In `deepeval`, this can be accomplished through **[G-Eval](/docs/metrics-llm-evals)**.
:::

Assessing Correctness involves comparing an LLM's actual output with the ground truth, but the process is not as straightforward as it may seem. There are important things to consider such as:

- Determining what constitutes your ground truth (selecting **evaluation parameters**)
- Defining the **evaluation steps/criteria** for assessing actual output against ground truth
- Establishing what constitutes an appropriate **threshold** to scale your correctness score

## How to create your Correctness Metric

### 1. Instantiate a `GEval` object

Begin creating your Correctness metric by instantiating a `GEval` object, choosing your evaluation LLM, and naming the metric accordingly.
"""
logger.info("## How to create your Correctness Metric")

correctness_metric = GEval(
    name="Correctness",
    model="llama3.2",
    ...
)

"""
:::tip
G-Eval is most effective when employing a model from the **GPT-4 model family** as your evaluation LLM, especially when it comes to assessing correctness.
:::

### 2. Select your evaluation parameters

G-Eval allows you to select parameters that are relevant for evaluation by providing a list of `LLMTestCaseParams`, which includes:

- `LLMTestCaseParams.INPUT`
- `LLMTestCaseParams.ACTUAL_OUTPUT`
- `LLMTestCaseParams.EXPECTED_OUTPUT`
- `LLMTestCaseParams.CONTEXT`
- `LLMTestCaseParams.RETRIEVAL_CONTEXT`

`ACTUAL_OUTPUT` should **always** be included in your `evaluation_params`, as this is what every Correctness metric will be directly evaluating. As mentioned earlier, Correctness is determined by how well the actual output aligns with the ground truth, which is typically more variable. The ground truth is best represented by `EXPECTED_OUTPUT`, where the expected output serves as the **ideal reference** for the actual output, with an exact match earning a score of 1.
"""
logger.info("### 2. Select your evaluation parameters")

correctness_metric = GEval(
    name="Correctness",
    model="llama3.2",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT],
    ...
)

"""
If the expected output is unavailable, you can alternatively compare the actual output with the `CONTEXT`, which serves as the **ideal retrieval context** for a RAG application. This comparison comes with its own set of evaluation criterias, however, which we will explore in the following step.
"""
logger.info("If the expected output is unavailable, you can alternatively compare the actual output with the `CONTEXT`, which serves as the **ideal retrieval context** for a RAG application. This comparison comes with its own set of evaluation criterias, however, which we will explore in the following step.")

correctness_metric = GEval(
    name="Correctness",
    model="llama3.2",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT],
    ...
)

"""
### 3. Defining your Evaluation Criteria

`G-Eval` lets you either provide a criteria from which it generates evaluation steps to assess your `evaluation_parameters`, or directly input the evaluation steps yourself. It's **always** recommended to supply your own `evaluation_steps` when building a custom Correctness metric, as this allows you to have **more control over how Correctness is defined**.

Here is a simple example of how one might define a basic Correctness metric:
"""
logger.info("### 3. Defining your Evaluation Criteria")

correctness_metric = GEval(
    name="Correctness",
    model="llama3.2",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

"""
Here's a more complex set of `evaluation_steps`, where detail is crucial to ensuring Correctness:
"""
logger.info("Here's a more complex set of `evaluation_steps`, where detail is crucial to ensuring Correctness:")

correctness_metric = GEval(
    name="Correctness",
    model="llama3.2",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
       'Compare the actual output directly with the expected output to verify factual accuracy.',
       'Check if all elements mentioned in the expected output are present and correctly represented in the actual output.',
       'Assess if there are any discrepancies in details, values, or information between the actual and expected outputs.'
    ],
)

"""
Here's another example metric which prioritizes general factual correctness over minutiae:
"""
logger.info("Here's another example metric which prioritizes general factual correctness over minutiae:")

correctness_metric = GEval(
    name="Correctness",
    model="llama3.2",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also lightly penalize omission of detail, and focus on the main idea",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
)

"""
Each evaluation dataset is unique, so it's important to iteratively **adjust your `evaluation_steps`** until your Correctness metric produces scores that align with your expectations. Whether this means giving more importance to detail, numerical values, structure, or even defining a new set of evaluation steps relative to the context instead of the expected output, is up for experimentation. The key is to **keep refining the metrics until they deliver the desired scores**.

:::note
G-Eval metrics remain relatively stable across multiple evaluations, despite the variability of LLM responses. Therefore, once you establish a satisfactory set of `evaluation_steps`, your Correctness metric should be **relatively robust**.
:::

**Congratulations 🎉!** You've just learnt how to build a Correctness metric for your custom LLM application. In the next section, we'll go over how to select an appropriate threshold for your Correctness metric.

## Iterating your `evaluations_steps`

You may wonder what it means to **iterate on your Correctness metric** until it aligns with your expectations. The answer is to have expectations! Once you establish an evaluation dataset and decide to assess your test cases for correctness, it's essential to establish a **baseline benchmark** by initially identifying which cases should score well and which should not, based on the needs of your LLM application.

Here is an example based on a detail-oriented Correctness metric:
"""
logger.info("## Iterating your `evaluations_steps`")


first_test_case = LLMTestCase(input="Summarize the benefits of daily exercise.",
                              actual_output="Daily exercise improves cardiovascular health, boosts mood, and enhances overall fitness.",
                              expected_output="Daily exercise improves cardiovascular health, boosts mood, and enhances overall fitness.")

second_test_case = LLMTestCase(input="Explain the process of photosynthesis.",
                               actual_output="Photosynthesis is how plants make their food using sunlight.",
                               expected_output="Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. It involves the green pigment chlorophyll and generates oxygen as a byproduct.")

third_test_case = LLMTestCase(input="Describe the effects of global warming.",
                              actual_output="Global warming leads to colder winters.",
                              expected_output="Global warming causes more extreme weather, including hotter summers, rising sea levels, and increased frequency of extreme weather events.")

test_cases = [first_test_test_case, second_test_case, third_test_case]

dataset = EvaluationDataset(test_cases=test_cases)

"""
Having a benchmark helps guide the development of your metric, and the primary method to align your evaluations with this baseline is by adjusting your `evaluation_steps`, as detailed in step 3 above.

# Finding the Right Threshold

You may initially achieve an 80% or even over 90% alignment with your expectations simply by tweaking the `evaluation_steps`. However, it's very **common to hit a plateau** at this stage. Identifying the correct threshold becomes essential at this point. It represents the crucial step in refining your custom metric to fully meet your expectations—and it's much simpler than you think!

### Step 1: Perform Correctness Evaluation

First, perform the Correctness evaluation on your dataset:
"""
logger.info("# Finding the Right Threshold")

correctness_metric = GEval(
    name="Correctness",
    model="llama3.2",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
        "Lightly penalize omissions of detail, focusing on the main idea",
        "Vague language or contradicting opinions are permissible"
    ],
)

deepeval.login("your_api_key_here")
dataset = EvaluationDataset()
dataset.pull(alias="dataset_for_correctness")

evaluation_output = dataset.evaluate([correctness_metric])

"""
### Step 2: Determine the Threshold

Next, determine the percentage of test cases you expect to be correct, extract all the test scores, and calculate the threshold accordingly:
"""
logger.info("### Step 2: Determine the Threshold")

scores = [output.metrics[0].score for output in evaluation_output]

def calculate_threshold(scores, percentile):
    sorted_scores = sorted(scores)
    index = int(len(sorted_scores) * (1 - percentile / 100))
    return sorted_scores[index]

percentile = 75  # Targeting the top 25%
threshold = calculate_threshold(scores, percentile)

"""
By following these steps, you can fine-tune the threshold to ensure your evaluation metrics align closely with your expectations, achieving the level of precision required for your specific needs.
"""
logger.info("By following these steps, you can fine-tune the threshold to ensure your evaluation metrics align closely with your expectations, achieving the level of precision required for your specific needs.")

logger.info("\n\n[DONE]", bright=True)