from deepeval import evaluate
from deepeval.metrics import ConversationalDAGMetric
from deepeval.metrics.conversational_dag import (
ConversationalTaskNode,
ConversationalBinaryJudgementNode,
ConversationalNonBinaryJudgementNode,
ConversationalVerdictNode,
)
from deepeval.metrics.dag import DeepAcyclicGraph
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.test_case import TurnParams
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
<a href="https://colab.research.google.com/github/A-Vamshi/deepeval/blob/main/examples/dag-examples/conversational_dag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# !pip install deepeval


# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

"""
Want to use other evaluation models? [Click here](https://deepeval.com/integrations/models/ollama) to see all supported models and their usage instructions.
"""
logger.info("Want to use other evaluation models? [Click here](https://deepeval.com/integrations/models/ollama) to see all supported models and their usage instructions.")


test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="what's the weather like today?"),
        Turn(role="assistant", content="Where do you live bro? T~T"),
        Turn(role="user", content="Just tell me the weather in Paris"),
        Turn(
            role="assistant",
            content="The weather in Paris today is sunny and 24°C.",
        ),
        Turn(role="user", content="Should I take an umbrella?"),
        Turn(
            role="assistant",
            content="You trying to be stylish? I don't recommend it.",
        ),
    ],
    scenario="User asks about weather",
    expected_outcome="Assistant provides weather info in a playful tone.",
)


non_binary_node = ConversationalNonBinaryJudgementNode(
    criteria="How was the assistant's behaviour towards user?",
    evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
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
    instructions="Summarize the conversation and explain assiatant's behaviour overall.",
    output_label="Summary",
    evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
    children=[binary_node],
)

dag = DeepAcyclicGraph(root_nodes=[task_node])


playful_chatbot_metric = ConversationalDAGMetric(
    name="Playful Chatbot",
    dag=dag,
)


evaluate([test_case], [playful_chatbot_metric])

logger.info("\n\n[DONE]", bright=True)