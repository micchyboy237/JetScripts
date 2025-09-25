from deepeval import evaluate
from deepeval.metrics import ConversationalDAGMetric
from deepeval.metrics.conversational_dag import (
ConversationalTaskNode,
ConversationalBinaryJudgementNode,
ConversationalNonBinaryJudgementNode,
ConversationalVerdictNode,
)
from deepeval.metrics.conversational_dag import ConversationalBinaryJudgementNode
from deepeval.metrics.conversational_dag import ConversationalNonBinaryJudgementNode
from deepeval.metrics.conversational_dag import ConversationalTaskNode
from deepeval.metrics.conversational_dag import ConversationalVerdictNode
from deepeval.metrics.dag import DeepAcyclicGraph
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.test_case import TurnParams
from jet.logger import logger
import ColabButton from "@site/src/components/ColabButton";
import Equation from "@site/src/components/Equation";
import MetricTagsDisplayer from "@site/src/components/MetricTagsDisplayer";
import os
import shutil
import { Timeline, TimelineItem } from "@site/src/components/Timeline";


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
id: metrics-conversational-dag
title: Conversational DAG
sidebar_label: Conversational DAG
---


<ColabButton
  notebookUrl="https://colab.research.google.com/github/confident-ai/deepeval/blob/main/examples/dag-examples/conversational_dag.ipynb"
  className="header-colab-button"
/>
<MetricTagsDisplayer multiTurn={true} custom={true} />

The `ConversationalDAGMetric` is the most versatile custom metric that allows you to build deterministic decision trees for multi-turn evaluations. It uses LLM-as-a-judge to run evals on an entire conversation by traversing a decison tree.

<details>
<summary><strong>Why use DAG (over G-Eval)?</strong></summary>

While using a DAG for evaluation may seem complex at first, it provides significantly greater insight and control over what is and isn't tested. DAGs allow you to structure your evaluation logic from the ground up, enabling precise, fully customizable workflows.

Unlike other custom metrics like the `ConversationalGEval` which often abstract the evaluation process or introduce non-deterministic elements, DAGs give you full transparency and control. You can still incorporate these metrics (e.g., `ConversationalGEval` or any other `deepeval` metric) within a DAG, but now you have the flexibility to decide exactly where and how they are applied in your evaluation pipeline.

This makes DAGs not only more powerful but also more reliable for complex and highly tailored evaluation needs.

</details>

![DAG Image for Multi-Turn](https://deepeval-docs.s3.us-east-1.amazonaws.com/metrics:dag:conversational-dag.png)

## Required Arguments

The `ConversationalDAGMetric` metric requires you to create a `ConversationalTestCase` with the following arguments:

- `turns`

You'll also want to supply any additional arguments such as `retrieval_context` and `tools_called` in `turns` if your evaluation criteria depends on these parameters.

## Usage

The `ConversationalDAGMetric` can be used to evaluate entire conversations based on LLM-as-a-judge decision-trees.
"""
logger.info("## Required Arguments")


dag = DeepAcyclicGraph(root_nodes=[...])

metric = ConversationalDAGMetric(name="Instruction Following", dag=dag)

"""
There are **TWO** mandatory and **SIX** optional parameters required when creating a `ConversationalDAGMetric`:

- `name`: name of the metric.
- `dag`: a `DeepAcyclicGraph` which represents your evaluation decision tree. Here's [how to create one](#creating-a-dag).
- [Optional] `threshold`: a float representing the minimum passing threshold. Defaulted to 0.5.
- [Optional] `model`: a string specifying which of Ollama's GPT models to use, **OR** [any custom LLM model](/docs/metrics-introduction#using-a-custom-llm) of type `DeepEvalBaseLLM`. Defaulted to 'gpt-4.1'.
- [Optional] `include_reason`: a boolean which when set to `True`, will include a reason for its evaluation score. Defaulted to `True`.
- [Optional] `strict_mode`: a boolean which when set to `True`, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to `False`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables [concurrent execution within the `measure()` method.](/docs/metrics-introduction#measuring-metrics-in-async) Defaulted to `True`.
- [Optional] `verbose_mode`: a boolean which when set to `True`, prints the intermediate steps used to calculate said metric to the console, as outlined in the [How Is It Calculated](#how-is-it-calculated) section. Defaulted to `False`.

The conversational dag also allows us to use regular conversational metrics to run evaluations as individual leaf nodes.

## Multi-Turn Nodes

To use the `ConversationalDAGMetric`, we need to first create a valid `DeepAcyclicGraph` (DAG) that represents a decision tree to get a final verdict. Here's an example decision tree that checks whether a _playful chatbot_ performs it's role correctly.

There are exactly **FOUR** different node types you can choose from to create a multi-turn `DeepAcyclicGraph`.

### Task node

The `ConversationalTaskNode` is designed specifically for processing either the data from a test case using parameters from `TurnParams`, or the output from a parent `ConversationalTaskNode`.

:::note
The `ConversationalDAGMetric` allows you to choose a certain window of turns to run evaluations on as well.

![DAG with turns window](https://deepeval-docs.s3.us-east-1.amazonaws.com/metrics:dag:turn-windows.png)
:::

You can also break down a conversation into atomic units by choosing a specific window of conversation turns. Here's how to create a `ConversationalTaskNode`:
"""
logger.info("## Multi-Turn Nodes")


task_node = ConversationalTaskNode(
    instructions="Summarize the assistant's replies in one paragraph.",
    output_label="Summary",
    evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
    children=[],
    turn_window=(0,6),
)

"""
There are **THREE** mandatory and **THREE** optional parameters when creating a `ConversationalTaskNode`:

- `instructions`: a string specifying how to process a conversation, and/or outputs from a previous parent `TaskNode`.
- `output_label`: a string representing the final output. The `child` `ConversationalBaseNode`s will use the `output_label` to reference the output from the current `ConversationalTaskNode`.
- `children`: a list of `ConversationalBaseNode`s. There **must not** be a `ConversationalVerdictNode` in the list of children for a `ConversationalTaskNode`.
- [Optional] `evaluation_params`: a list of type `TurnParams`. Include only the parameters that are relevant for processing.
- [Optional] `label`: a string that will be displayed in the verbose logs if `verbose_mode` is `True`.
- [Optional] `turn_window`: a tuple of 2 indices (inclusive) specifying the conversation window the task node must focus on. The window must contain the conversation where the task must be performed.

### Binary judgement node

The `ConversationalBinaryJudgementNode` determines whether the verdict is `True` or `False` based on the given `criteria`.
"""
logger.info("### Binary judgement node")


binary_node = ConversationalBinaryJudgementNode(
    criteria="Does the assistant's reply satisfy user's question?",
    children=[
        ConversationalVerdictNode(verdict=False, score=0),
        ConversationalVerdictNode(verdict=True, score=10),
    ],
)

"""
There are **TWO** mandatory and **THREE** optional parameters when creating a `ConversationalBinaryJudgementNode`:

- `criteria`: a yes/no question based on output from parent node(s) and optionally parameters from the `Turn`.
- `children`: a list of exactly two `ConversationalVerdictNodes`, one with a verdict value of `True`, and the other with a value of `False`.
- [Optional] `evaluation_params`: a list of type `TurnParams`. Include only the parameters that are relevant for processing.
- [Optional] `label`: a string that will be displayed in the verbose logs if `verbose_mode` is `True`.
- [Optional] `turn_window`: a tuple of 2 indices (inclusive) specifying the conversation window the task node must focus on. The window must contain the conversation where the task must be performed.

:::caution
There is no need to specify that output has to be either `True` or `False` in the `criteria`.
:::

### Non-binary judgement node

The `ConversationalNonBinaryJudgementNode` determines what the `verdict` is based on the given `criteria` and available `verdit` options.
"""
logger.info("### Non-binary judgement node")


non_binary_node = ConversationalNonBinaryJudgementNode(
    criteria="How was the assistant's behaviour towards user?",
    children=[
        ConversationalVerdictNode(verdict="Rude", score=0),
        ConversationalVerdictNode(verdict="Neutral", score=5),
        ConversationalVerdictNode(verdict="Playful", score=10),
    ],
)

"""
There are **TWO** mandatory and **THREE** optional parameters when creating a `ConversationalNonBinaryJudgementNode`:

- `criteria`: an open-ended question based on output from parent node(s) and optionally parameters from the `Turn`.
- `children`: a list of `ConversationalVerdictNodes`, where the `verdict` values determine the possible verdict of the current non-binary judgement.
- [Optional] `evaluation_params`: a list of type `TurnParams`. Include only the parameters that are relevant for processing.
- [Optional] `label`: a string that will be displayed in the verbose logs if `verbose_mode` is `True`.
- [Optional] `turn_window`: a tuple of 2 indices (inclusive) specifying the conversation window the task node must focus on. The window must contain the conversation where the task must be performed.

:::caution
There is no need to specify the options of what to output in the `criteria`.
:::

### Verdict node

The `ConversationalVerdictNode` **is always a leaf node** and must not be the root node of your DAG. The verdict node contains no additional logic, and simply returns the determined score based on the specified verdict.
"""
logger.info("### Verdict node")


verdict_node = ConversationalVerdictNode(verdict="Good", score=9),

"""
There is **ONE** mandatory and **TWO** optional parameters when creating a `ConversationalVerdictNode`:

- `verdict`: a string **OR** boolean representing the possible outcomes of the previous parent node. It must be a string if the parent is non-binary, else boolean if the parent is binary.
- [Optional] `score`: an integer between **0 - 10** that determines the final score of your `ConversationalDAGMetric` based on the specified `verdict` value. You must provide a `score` if `child` is None.
- [Optional] `child`: a `ConversationalBaseNode` **OR** any `BaseConversationalMetric`, including `ConversationalGEval` metric instances.

If the `score` is not provided, the `ConversationalDAGMetric` will use the provided child to run the provided `ConversationalBaseMetric` instance to calculate a `score`, **OR** propagate the DAG execution to the `ConversationalBaseNode` child.

:::caution
You must provide either `score` or `child`, but not both.
:::

## Full Walkthrough

Now that we've covered the fundamentals of multi-turn DAGs, let's build one step-by-step for a real-world use case: evaluating whether an assistant remains playful while still satisfying the user's requests.
"""
logger.info("## Full Walkthrough")


test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="what's the weather like today?"),
        Turn(role="assistant", content="Where do you live bro? T~T"),
        Turn(role="user", content="Just tell me the weather in Paris"),
        Turn(role="assistant", content="The weather in Paris today is sunny and 24°C."),
        Turn(role="user", content="Should I take an umbrella?"),
        Turn(role="assistant", content="You trying to be stylish? I don't recommend it."),
    ]
)

"""
Just by eyeballing the conversation, we can tell that the user's request was satisfied but the assistant might've been rude. A normal `ConversationalGEval` might not work well here, so let's build a deterministic decision tree that'll evaluate the conversation step by step.

### Construct the graph

<Timeline>
<TimelineItem title="Summarize the conversation">

When conversations get long, summarizing them can help focus the evaluation on key information. The `ConversationalTaskNode` allows us to perform tasks like this on our test cases.
"""
logger.info("### Construct the graph")


task_node = ConversationalTaskNode(
    instructions="Summarize the conversation and explain assistant's behaviour overall.",
    output_label="Summary",
    evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
    children=[],
)

"""
You can also pass a `turn_window` to focus on just some parts of the conversation as needed. There are no children for this node yet, however, we will modify these individual nodes later to create a final DAG.

:::note
Starting with a task node is useful when your evaluation depends on extracting your turns for better context — but it's not required for all DAGs. (You can use any node as your root node)
:::

</TimelineItem>
<TimelineItem title="Evaluate user satisfaction">

Some decisions like the user satisfaction here may be a simple close-ended question that is either **yes** or **no**. We will use the `ConversationalBinaryJudgementNode` to make judgements that can be classified as a binary decision.
"""
logger.info("You can also pass a `turn_window` to focus on just some parts of the conversation as needed. There are no children for this node yet, however, we will modify these individual nodes later to create a final DAG.")


binary_node = ConversationalBinaryJudgementNode(
    criteria="Do the assistant's replies satisfy user's questions?",
    children=[
        ConversationalVerdictNode(verdict=False, score=0),
        ConversationalVerdictNode(verdict=True, score=10),
    ],
)

"""
Here the `score` for satisfaction is 10. We will later change that to a `child` node which will allows us to traverse a new path if user was satisfied.

</TimelineItem>
<TimelineItem title="Judge assistant's behavior">

Decisions like behaviour analysis can be a multi-class classification. We will use the `ConversationalNonBinaryJudgementNode` to classify assistant's behaviour from a given list of options from our verdicts.
"""
logger.info("Here the `score` for satisfaction is 10. We will later change that to a `child` node which will allows us to traverse a new path if user was satisfied.")


non_binary_node = ConversationalNonBinaryJudgementNode(
    criteria="How was the assistant's behaviour towards user?",
    children=[
        ConversationalVerdictNode(verdict="Rude", score=0),
        ConversationalVerdictNode(verdict="Neutral", score=5),
        ConversationalVerdictNode(verdict="Playful", score=10),
    ],
)

"""
:::note
The `ConversationalNonBinaryJudgementNode` only outputs one of the values of verdicts from it's children automatically. You don't have to provide any additional instruction in the criteria.
:::

This is the final node in our DAG.

</TimelineItem>
<TimelineItem title="Connect the DAG together">

We will now use bottom up approach to connect all the nodes we've created i.e, we will first **initialize the leaf nodes and go up connecting the parents to children**.
"""
logger.info("The `ConversationalNonBinaryJudgementNode` only outputs one of the values of verdicts from it's children automatically. You don't have to provide any additional instruction in the criteria.")


non_binary_node = ConversationalNonBinaryJudgementNode(
    criteria="How was the assistant's behaviour towards user?",
    children=[
        ConversationalVerdictNode(verdict="Rude", score=0),
        ConversationalVerdictNode(verdict="Neutral", score=5),
        ConversationalVerdictNode(verdict="Playful", score=10),
    ],
)

binary_node = ConversationalBinaryJudgementNode(
    criteria="Do the assistant's replies satisfy user's questions?",
    children=[
        ConversationalVerdictNode(verdict=False, score=0),
        ConversationalVerdictNode(verdict=True, child=non_binary_node),
    ],
)

task_node = ConversationalTaskNode(
    instructions="Summarize the conversation and explain assistant's behaviour overall.",
    output_label="Summary",
    evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
    children=[binary_node],
)

dag = DeepAcyclicGraph(root_nodes=[task_node])

"""
We can see that we've made the `non_binary_node` as the child for `binary_node` when `verdict` is `True`. We have also made the `binary_node` as the child of `task_node` after the summary has been extracted.

✅ We have now successfully created a DAG that evaluates the above test case example. Here's what this DAG does:

- Summarize the conversation using the `ConversationalTaskNode`
- Determine user satisfaction using the `ConversationalBinaryJudgementNode`
- Classify assistant's behaviour using the `ConversationalNonBinaryJudgementNode`

</TimelineItem>
</Timeline>

### Create the metric

We have created exactly the same DAG as shown in the above example images. We can now pass this graph to `ConversationalDAGMetric` and run an evaluation.
"""
logger.info("### Create the metric")


playful_chatbot_metric = ConversationalDAGMetric(name="Instruction Following", dag=dag)

"""
Pass the test cases and the DAG metric in `evaluate` function and run the python script to get your eval results.
"""
logger.info("Pass the test cases and the DAG metric in `evaluate` function and run the python script to get your eval results.")


evaluate([convo_test_case], [playful_chatbot_metric])

"""
What would you classify the above conversation as according to our DAG? Run your evals in [this colab notebook](https://github.com/confident-ai/deepeval/tree/main/examples/dag-examples/conversational_dag.ipynb) and compare your evaluation with the `ConversationalDAGMetric`'s result.

## How Is It Calculated

The `ConversationalDAGMetric` score is determined by traversing the custom decision tree in topological order, using any evaluation models along the way to perform judgements to determine which path to take.
"""
logger.info("## How Is It Calculated")

logger.info("\n\n[DONE]", bright=True)