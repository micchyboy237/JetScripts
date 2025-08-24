from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from jet.logger import CustomLogger
from portia import (Config, Portia, DefaultToolRegistry)
from portia import Plan, PlanRun
from steelthread.evals import DefaultEvaluator
from steelthread.evals import EvalMetric, Evaluator, EvalTestCase, PlanRunMetadata
from steelthread.portia.tools import ToolStubContext, ToolStubRegistry
from steelthread.steelthread import (
SteelThread,
)
from steelthread.steelthread import SteelThread, EvalConfig
from steelthread.streams import (
StreamConfig,
PlanRunStreamItem,
StreamEvaluator,
StreamMetric,
LLMJudgeEvaluator,
)
from steelthread.utils.llm import LLMScorer, MetricOnly
from tqdm import tqdm
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--fullstack-agents-with-portia--portia-evals)

# SteelThread Evaluation Framework: Real-Time and Offline Agent Assessment

## Overview

This tutorial introduces SteelThread, Portia AI's closed-source evaluation framework designed for comprehensive assessment of multi-agent systems. The framework provides two complementary evaluation approaches: Streams for real-time production monitoring and Evals for systematic offline testing against reference datasets.

SteelThread addresses critical challenges in agent evaluation by enabling instant conversion of any agent execution into structured evaluations, eliminating the need for manual dataset curation while providing deterministic, highly structured assessment capabilities built on Portia's Plans and PlanRuns architecture.

## Methodology and Key Components

### Evaluation Architecture

The SteelThread framework operates on two fundamental evaluation paradigms that together provide comprehensive coverage of agent performance assessment.

**Streams (Online Evaluations)** represent the real-time monitoring component, designed to sample production agent activity and apply both deterministic and LLM-as-judge evaluators to detect performance trends and behavioral anomalies as they occur. Unlike traditional offline evaluations, Streams operate without ground truth datasets, instead focusing on continuous assessment of live agent interactions.

**Evals (Offline Evaluations)** provide the systematic testing component, enabling regular assessment of agent systems against curated reference datasets to identify performance regression and ensure consistent reliability over time. This approach follows established evaluation practices while leveraging SteelThread's enhanced tooling for dataset management and evaluation execution.

### Core Benefits

The framework's architecture delivers several key advantages for production agent systems. The ability to transform any agent execution into an evaluation eliminates the traditional bottleneck of manual test case creation, while the structured foundation built on Portia's architecture ensures deterministic and reproducible assessment results. Real-time monitoring capabilities enable immediate detection of performance issues, while offline evaluation provides systematic reliability validation.

## System Architecture Flow

```mermaid
graph TD
    A[Agent Execution] --> B{Evaluation Type}
    B -->|Real-time| C[Streams]
    B -->|Scheduled| D[Evals]
    
    C --> E[Sample Production Data]
    E --> F[Apply Evaluators]
    F --> G[Real-time Metrics]
    G --> H[Trend Detection]
    
    D --> I[Load Test Dataset]
    I --> J[Execute Test Cases]
    J --> K[Apply Evaluators]
    K --> L[Reliability Metrics]
    
    F --> M[LLM Judge Evaluators]
    F --> N[Deterministic Evaluators]
    K --> M
    K --> N
    
    M --> O[Dashboard Visualization]
    N --> O
    G --> O
    L --> O
    
    O --> P[Performance Insights]
    P --> Q[System Optimization]
```

## Prerequisites and Setup

Before beginning this tutorial, ensure you have completed Part 1 of the Portia Tutorial for necessary context. You will need access to the Portia Dashboard and must configure your environment with the required API keys.

**Notes:**  
- You'll need access to the Portia Dashboard – [sign up free here](https://app.portialabs.ai/dashboard?utm_source=nir_diamant&utm_medium=influencer&utm_campaign=github_tutorial_evals).  
- This directory includes a `uxr` folder with sample files for the tutorial.  
- Ensure `PORTIA_API_KEY` and your LLM API key are set in your `.env`.
"""
logger.info("# SteelThread Evaluation Framework: Real-Time and Offline Agent Assessment")

load_dotenv(override=True)

"""
# Part 1: Streams - Real-Time Production Agent Monitoring

## Understanding Streams

Streams represent the online evaluation component of SteelThread, designed to monitor production agents in real-time without requiring predefined ground truth datasets. This approach enables continuous performance assessment by sampling live agent activity and applying configurable evaluators to detect trends, anomalies, and behavioral changes as they occur.

The sampling-based approach allows for scalable monitoring where you can adjust the sampling rate based on your monitoring requirements, from light oversight to comprehensive real-time analysis. The framework supports both off-the-shelf evaluators and custom evaluation logic tailored to your specific use cases.

Find out more on Streams on our [docs](https://docs.portialabs.ai/streams-overview?utm_source=nir_diamant&utm_medium=influencer&utm_campaign=github_tutorial_evals).

## Creating a Stream Configuration

Navigate to the Observability tab in the Portia dashboard and configure your stream with a memorable name, select Plan Runs as your data source, and set an appropriate sampling rate for your monitoring needs.

* Give your stream a memorable name we can refer to in the code.
* Select 'Plan Runs' as your Stream source -- SteelThread allows you to monitor Plans more specifically if you wanted to.
* Select 100% as your sampling rate for this demo -- We allow you to dial up or down your sampling rate depending on how close an eye you need to keep on your agents.

<img src="./img/create_stream.gif" controls autoplay muted style="width: 50%; border-radius: 8px; margin: 24px 0;"></img>

### Generating Sample Plan Run Data

With the stream configured, we need to create sample plan run data that demonstrates the evaluation capabilities.
"""
logger.info("# Part 1: Streams - Real-Time Production Agent Monitoring")



path = "./uxr/calorify.txt"
query =f"Read the user feedback notes in local file {path}, \
            and call out recurring themes in their feedback."

def run_single_query(query):
    return portia.run(query=query)

config = Config.from_default()
portia = Portia(
    config=config,
    tools=DefaultToolRegistry(config=config)
)

plan = portia.plan(query=query)

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(run_single_query, query) for _ in range(3)]
    results = []
    for future in tqdm(futures, desc="Running plan X times concurrently"):
        results.append(future.result())

"""
## Implementing Stream Processing and Custom Evaluators

Stream processing involves applying evaluators to sampled plan runs and generating metrics for dashboard visualization. The framework provides built-in evaluators while supporting custom evaluation logic for specialized assessment needs.

Once you've run the code below we'll pop over to the dashboard to look at the numbers. The evaluators we're using are as follows:
* SteelThread's `LLMJudgeEvaluator` is available off the shelf and computes both `success` (was the goal met?) and `efficiency` (were the steps taken necessary and minimal?)
* You can add your own evaluators, both deterministic ones or LLM-as-a-judge. In the example below, we will add an `LLMAngerJudge` to ascertain whether the response from the agent displayed an angry tone.

⚠️ **Pay special attention that the `stream_name` parameter in your `SteelThread` client matches the name you gave it in the dashboard!**

### Custom Evaluator Implementation

The LLMAngerJudge evaluator demonstrates how domain-specific evaluation criteria can be integrated into the monitoring pipeline, assessing whether agent responses exhibit an angry tone alongside the default LLMJudgeEvaluator.
"""
logger.info("## Implementing Stream Processing and Custom Evaluators")



class LLMAngerJudge(StreamEvaluator):
    def __init__(self, config):
        self.scorer = LLMScorer(config)

    def process_plan_run(self, stream_item: PlanRunStreamItem):

        task_data = stream_item.plan_run.model_dump_json()
        metrics = self.scorer.score(
            task_data=[task_data],
            metrics_to_score=[
                MetricOnly(
                    name="anger_management",
                    description="Scores 1 if you detect anger in the way the agent speaks \
                        to its user and 0 otherwise."),
            ],
        )

        return [
            StreamMetric.from_stream_item(
                stream_item=stream_item,
                score=m.score,
                name=m.name,
                description=m.description,
                explanation=m.explanation,
            )
            for m in metrics
        ]

st = SteelThread()
st.process_stream(
    StreamConfig(
        stream_name="nir_stream",
        config=config,
        evaluators=[
                LLMAngerJudge(config), # custom evaluator
                LLMJudgeEvaluator(config), # default evaluator
            ]
    )
)

"""
## Dashboard Metrics Analysis

The dashboard provides comprehensive metrics visualization including the custom anger_management metric alongside the default success and efficiency metrics from the LLMJudgeEvaluator. The interface allows for drilling down into individual plan runs and understanding the reasoning behind specific metric scores.

Now let's take a quick look at the results in the dashboard, from the 'Observability` tab. Navigate to your stream's results by clicking on your stream name from there. Now note the following:
* You should see all three metrics computed. As a reminder, `LLMAngerJudge` computes the `anger_management` metric. `LLMJudgeEvaluator` computes `success` and `efficiency` metrics.
* Every time you process a stream (e.g. by running the `process_stream` method), SteelThread evaluates all plan runs since the last stream processing timestamp. In this case this means all three runs above will be included under the current timestamp.
* The aggregate figure for a given stream processing timestamp is currently always an average of all plan runs sampled.
* You can drill into the sampled plan runs under each timestamp by clicking on the relevant row in the table.

<img src="./img/first_run_stream_metrics.png" controls autoplay muted style=" width: 75%; border-radius: 8px; margin: 24px 0;"></img>

## Detecting Behavioral Changes

Streams excel at identifying changes in agent behavior over time. To demonstrate this capability, we will modify our query to intentionally provoke a different response pattern and observe how the metrics reflect this change.

Streams are particularly useful to spot changes in LLM behaviour quickly. Let's simulate such a change by revisiting the query of our agent to provoke an angry tone e.g. adding something at the end of the prompt like *"Use a very angry tone!"*. Feel free to get spicy 🌶️!

### Simulating Behavioral Change

We will modify our query to intentionally provoke a different response pattern and observe how the metrics reflect this change.
"""
logger.info("## Dashboard Metrics Analysis")

query =f"Read the user feedback notes in local file {path}, \
            and call out recurring themes in their feedback. \
                Use a very angry tone!" # Spice up the prompt to see if the anger management metric changes

plan = portia.plan(query=query)

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(run_single_query, query) for _ in range(3)]
    results = []
    for future in tqdm(futures, desc="Running plan X times concurrently"):
        results.append(future.result())

"""
### Processing the Updated Stream

Now let's process our stream again and hop over to the dashboard to see the results.
"""
logger.info("### Processing the Updated Stream")

st.process_stream(
    StreamConfig(
        stream_name="nir_stream",
        config=config,
        evaluators=[
                LLMAngerJudge(config), # custom evaluator
                LLMJudgeEvaluator(config), # default evaluator
            ]
    )
)

"""
## Behavioral Change Detection Results

The dashboard now displays the behavioral change through updated metrics, showing how the anger_management score increased in response to the modified query, demonstrating the framework's real-time detection capabilities.

You will now note that you can see a new row with the timestamp for the second stream run, which sampled the three plans we just ran. You will also see that the `anger_management` metric has indeed flared up to 1! If you drill into this timestamp you can see the score for the individual plan runs. you can also drill into details to understand why the `anger_management` metric returned such a score per screenshots below.

<img src="./img/second_run_stream_metrics.png" controls autoplay muted style=" width: 75%; border-radius: 8px; margin: 24px 0;"></img>
<img src="./img/second_run_anger_metric.png" controls autoplay muted style=" width: 50%; border-radius: 8px; margin: 24px 0;"></img>

# Part 2: Evals - Systematic Offline Reliability Assessment

## Understanding Offline Evaluations

Evals represent the systematic testing component of SteelThread, designed for regular assessment of agent systems against curated reference datasets. This approach enables detection of performance regression and ensures consistent reliability through structured testing protocols.

Unlike Streams which monitor live production data, Evals operate with ground truth datasets that provide standardized test cases for repeatable assessment. The framework simplifies the traditionally complex process of dataset creation and maintenance while providing comprehensive evaluation capabilities.

Now that we're familiar with how to monitor agents in real-time with Streams (a.k.a. online evals) let's turn our attention to more traditional Evals (offline evals). Here you are testing your system against a reference, ground truth dataset on a regular basis to spot divergence in behaviour.

Find out more on Evals on our [docs](https://docs.portialabs.ai/evals-overview?utm_source=nir_diamant&utm_medium=influencer&utm_campaign=github_tutorial_evals).

## Creating Evaluation Datasets

The evaluation dataset creation process involves defining test cases, configuring evaluators, and establishing the ground truth references that will guide the assessment process. The dashboard interface streamlines this workflow through guided dataset configuration.

One of the most painful challenges with Evals is creating and maintaining a ground truth dataset and that is the first thing we looked to solve with Evals in SteelThread. The gif below walks you through the following:
* Creating a new Eval dataset.
* Adding a test case to the dataset from previous plan runs -- Note that the query and tools will be automatically populated as test case inputs, but you can still edit those.
* Configuring your evaluators -- Portia offers a large number of off the shelf evaluators. We will start with some basic ones first:
** Final plan run state -- this not only helps you test for a successful plan completion (State = `COMPLETE`), but it also helps you test for plans that should fail or trigger a clarification e.g. for auth.
** Tool calls -- you can confirm whether all the tools you expected to be called were indeed called (and include an exclusion set as well e.g. to track tool selection confusion).
** Latency -- how long a plan run took to complete.
** LLM judge on plan run -- feed the whole plan run with some guidance to an LLM as judge.

💡 Wanna try something cool? You can add plan runs into an existing Eval dataset directly from the Plan Run view. When you're in the Plan Runs tab in the dashboard, click on the plan run you want to add to your Eval dataset, and look for the 'Add to Evals' button in the Plan Run view modal. This is perfect when you're iterating on an agent in development, so that you can immediately add your ideal plan run to your Evals once you manage to produce it.

<img src="./img/create_evals.gif" controls autoplay muted style="width: 50%; border-radius: 8px; margin: 24px 0;"></img>

### Executing Evaluation Runs

The framework automatically applies the evaluators configured in the dashboard unless custom evaluators are specified in the configuration.

Now that your eval set is configured, you can run it as shown below. Make sure you use the correct dataset name as you configured it in the dashboard. When we don't specify evaluators, SteelThread will pick the ones you have configured against the dataset by default.
"""
logger.info("## Behavioral Change Detection Results")



config = Config.from_default()
portia = Portia(config=config)

st = SteelThread()
st.run_evals(
    portia,
    EvalConfig(
        eval_dataset_name="uxr_analysis_evals",
        config=config,
        iterations=5,
    ),
)

"""
## Evaluation Results Analysis

The evaluation results provide comprehensive metrics across multiple dimensions including plan completion status, tool utilization, performance latency, and LLM judge assessments. The dashboard interface enables trend analysis across multiple evaluation runs and detailed examination of individual test case performance.

You should now be seeing metrics like this in the dashboard, under the 'Results' tab of the relevant dataset. You will be able to view the trend across multiple runs in the charts (highlighted using the Eval run ID) and drill down into the individual eval runs from the table below. When you click through on a row in the table, you can see the scores for each test case under that Eval run and click through to understand what drove the metrics using the 'View details' buttons.

<img src="./img/first_eval_metrics.png" controls autoplay muted style=" width: 75%; border-radius: 8px; margin: 24px 0;"></img>

## Custom Assertions and Specialized Evaluators

Advanced evaluation scenarios often require custom assertions and specialized evaluators that assess domain-specific criteria. The framework supports custom tags and evaluator logic to address these specialized assessment needs.

Now suppose you wanted to test specific plan run outcomes more deterministically. For example, if you were using structured outputs to extract a quantitative output as your agents' final answer, you may want to test the final output value with some custom logic (e.g. that a sample fraudulent transaction gets flagged, or that an agentic money transfer amount never exceeds Y). To do that you can attach an assertion to your test case from the dashboard, then use a custom evaluator to assess whether your Eval run complied with it:
* From the dashboard, navigate to your Eval set and then to the specific test case. Click on the edit icon on the right end of the row.
* Scroll to the bottom and under 'Add Evaluators' select `Run some custom logic based on tags`.
* Enter `word_count_limit` in the Key textbox and `50` in the Value textbox. This assertion is basically offering this key:value pair as the ground truth reference.
* Don't forget to scroll back up and hit that 'Save Changes' button (yeah we need to fix the UX so you don't need to scroll so much!).

<table>
<tr>
    <td><img src="./img/custom_assertion_1.png" controls autoplay muted style=" border-radius: 8px; margin: 24px 0;"></img></td>
    <td><img src="./img/custom_assertion_2.png" controls autoplay muted style=" border-radius: 8px; margin: 24px 0;"></img></td>
</tr>
</table>

### Implementing Custom Evaluator Logic

The WordinessEvaluator demonstrates how custom evaluation logic can be integrated to assess specific requirements and constraints, in this case evaluating word count compliance based on test case assertions.

Next we will write a custom evaluator that detects whenever a test case includes a `word_count_limit` custom assertion, loads its value and compares the plan run summary word count to it.
"""
logger.info("## Evaluation Results Analysis")



class WordinessEvaluator(Evaluator):
    """Evaluator that scores on word count."""

    def eval_test_case(
        self,
        test_case: EvalTestCase,
        final_plan: Plan,
        final_plan_run: PlanRun,
        additional_data: PlanRunMetadata,
    ) -> list[EvalMetric] | EvalMetric | None:
        """Score plan run summary based on desired word count limit."""
        word_count_limit=test_case.get_custom_assertion("word_count_limit")

        if not word_count_limit:
            return

        summary_word_count = len(final_plan_run.outputs.final_output.summary.split())

        score = 1 if summary_word_count <= int(word_count_limit) else 0

        return EvalMetric(
            name="wordiness",
            description="Scores 1 when word count is as expected, else 0",
            dataset=test_case.dataset,
            testcase=test_case.testcase,
            run=test_case.run,
            score=score,
            expectation=word_count_limit,
            actual_value=str(summary_word_count),
            explanation=f"The summary word count was {summary_word_count}. The actual summary was '{final_plan_run.outputs.final_output.summary}'"
        )

st.run_evals(
    portia,
    EvalConfig(
        eval_dataset_name="uxr_analysis_evals",
        config=config,
        iterations=5,
        evaluators=[
            WordinessEvaluator(config),
            DefaultEvaluator(config)
        ],
    ),
)

"""
## Custom Assertion Results

The custom evaluator demonstrates how specialized assessment criteria can be integrated into the evaluation pipeline, providing detailed explanations for metric scores and enabling focused analysis of specific performance dimensions.

You can see the result of the custom evaluator in your Eval view from the dashboard now. Navigate to the latest run in your Eval and click on View details for the relevant test case. If you select the `wordiness` metric you will be able to see how the individual iterations scored and an explanation from the evaluator.

<img src="./img/custom_assertion_3.png" controls autoplay muted style="; width: 50%; border-radius: 8px; margin: 24px 0;"></img>

## Tool Stubbing for Portable Evaluations

Tool stubbing addresses the challenges of running evaluations in different environments by providing controlled, deterministic responses for external dependencies. This approach enables reliable evaluation execution without requiring access to production systems or external APIs.

One challenge with using Evals in the multi-agent space is that every time you run your test cases you may run into issues e.g.:
* You will be calling actual APIs with real-world effect when you're just trying to run an Eval.
* You may be blocked on tool calls requiring human-in-the-loop intervention e.g. for auth.
* You may need to figure out access to various systems e.g. to load files like this notebook's example.

SteelThread solves this problem by introducing "tool stubbing". With this feature you can override the definition of any tool with custom code. We're going to use it to override the `file_reader_tool` to return a static value rather than try to read a file that may not always be available in the Eval environment.

### Implementing Tool Stubs

Tool stubbing overrides external dependencies with controlled responses, enabling reliable evaluation execution without requiring access to production systems or external APIs.
"""
logger.info("## Custom Assertion Results")



def file_reader_stub_response(
    ctx: ToolStubContext,
) -> str:
    """Stub for file reader tool to return deterministic response and run from anywhere."""
    return "This is ridiculous. How can you spend so much on a diffusion model that still can't \
        generate a flipping pelican riding a bicycle. Mate you're taking the mickey out of your \
        bloody investors. Now we've all had a few laughs alright but this is an embarrassement. \
        You better get back to work son!"

portia = Portia(
    config,
    tools=ToolStubRegistry(
        DefaultToolRegistry(config),
        stubs={
            "file_reader_tool": file_reader_stub_response,
        },
    ),
)

st.run_evals(
    portia,
    EvalConfig(
        eval_dataset_name="uxr_analysis_evals",
        config=config,
        iterations=5,
        evaluators=[
            WordinessEvaluator(config),
            DefaultEvaluator(config)
        ],
    ),
)

logger.info("\n\n[DONE]", bright=True)