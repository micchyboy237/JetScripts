from crewai import Task, Crew, Agent
from deepeval.dataset import EvaluationDataset
from deepeval.integrations.crewai import instrument_crewai
from deepeval.metrics import AnswerRelevancyMetric
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
## Evaluating CrewAI's `crew` (end-to-end)

In this notebook we will demonstrate how you can run evaluations on crews using datasets from Confident AI and DeepEval's dataset iterator.

### Install dependencies:
"""
logger.info("## Evaluating CrewAI's `crew` (end-to-end)")

# !pip install -U deepeval -U crewai ipywidgets --quiet

"""
### Set your Ollama API key:
"""
logger.info("### Set your Ollama API key:")


# os.environ["OPENAI_API_KEY"] = "<your-ollama-api-key>"

"""
### Create a crew:

This is a simple crew with a single agent and a single task.
"""
logger.info("### Create a crew:")


agent = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
)

task = Task(
    description="Explain the given topic: {topic}",
    expected_output="A clear and concise explanation.",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task])

result = crew.kickoff(
    inputs={"topic": "What is the biggest open source database?"}
)
logger.debug(result)

"""
### Evaluate the agent

To evaluate CrewAI's `crew`:

1. Instrument the application (using `from deepeval.integrations.crewai import instrument_crewai`)
2. Supply metrics to `kickoff`.


> (Pro Tip) View your Agent's trace and publish test runs on [Confident AI](https://www.confident-ai.com/). Apart from this you get an in-house dataset editor and more advaced tools to monitor and enventually improve your Agent's performance. Get your API key from [here](https://app.confident-ai.com/)
"""
logger.info("### Evaluate the agent")

os.environ["CONFIDENT_API_KEY"] = "<your-confident-api-key>"


instrument_crewai()

"""
### Using a dataset from Confident AI:

For demo purposes, we will use a public dataset from Confident AI. You can use your own dataset as well. Refer to the [docs](https://deepeval.com/docs/evaluation-end-to-end-llm-evals#setup-your-test-environment) to learn more about how to create your own dataset.
"""
logger.info("### Using a dataset from Confident AI:")


dataset = EvaluationDataset()
dataset.pull(alias="topic_agent_queries", public=True)

"""
### Run evaluations:

We will use the `AnswerRelevancyMetric` to evaluate the crew. Dataset iterator will yield golden examples from the dataset.
"""
logger.info("### Run evaluations:")


for golden in dataset.evals_iterator():
    result = crew.kickoff(
        inputs={"topic": golden.input}, metrics=[AnswerRelevancyMetric()]
    )

"""
Congratulation! You have just evaluated your first CrewAI's `crew` using Deepeval. Try changing Hyperparameters, Agents, Tasks, Metrics and see how your agent performs.
"""
logger.info("Congratulation! You have just evaluated your first CrewAI's `crew` using Deepeval. Try changing Hyperparameters, Agents, Tasks, Metrics and see how your agent performs.")

logger.info("\n\n[DONE]", bright=True)