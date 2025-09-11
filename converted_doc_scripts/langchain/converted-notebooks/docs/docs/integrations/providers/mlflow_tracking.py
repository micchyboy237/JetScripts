from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import mlflow
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
# MLflow

>[MLflow](https://mlflow.org/) is a versatile, open-source platform for managing workflows and artifacts across the machine learning and generative AI lifecycle. It has built-in integrations with many popular AI and ML libraries, but can be used with any library, algorithm, or deployment tool.

MLflow's [LangChain integration](https://mlflow.org/docs/latest/llms/langchain/autologging.html) provides the following capabilities:

- **[Tracing](https://mlflow.org/docs/latest/llms/langchain/autologging.html)**: Visualize data flows through your LangChain components with one line of code (`mlflow.langchain.autolog()`)
- **[Experiment Tracking](https://mlflow.org/docs/latest/llms/langchain/index.html#experiment-tracking)**: Log artifacts, code, and metrics from your LangChain runs
- **[Model Management](https://mlflow.org/docs/latest/model-registry.html)**: Version and deploy LangChain applications with dependency tracking
- **[Evaluation](https://mlflow.org/docs/latest/llms/langchain/index.html#mlflow-evaluate)**: Measure the performance of your LangChain applications

**Note**: MLflow tracing is available in MLflow versions 2.14.0 and later.

This short guide focuses on MLflow's tracing capability for LangChain and LangGraph applications. You'll see how to enable tracing with one line of code and view the execution flow of your applications. For information about MLflow's other capabilities and to explore additional tutorials, please refer to the [MLflow documentation for LangChain](https://mlflow.org/docs/latest/llms/langchain/index.html). If you're new to MLflow, check out the [Getting Started with MLflow](https://mlflow.org/docs/latest/getting-started/index.html) guide.

## Setup

To get started with MLflow tracing for LangChain, install the MLflow Python package. We will also use the `langchain-ollama` package.
"""
logger.info("# MLflow")

# %pip install mlflow langchain-ollama langgraph -qU

"""
Next, set the MLflow tracking URI and Ollama API key.
"""
logger.info("Next, set the MLflow tracking URI and Ollama API key.")


os.environ["MLFLOW_TRACKING_URI"] = ""
# os.environ["OPENAI_API_KEY"] = ""

"""
## MLflow Tracing

MLflow's tracing capability helps you visualize the execution flow of your LangChain applications. Here's how to enable it.
"""
logger.info("## MLflow Tracing")


mlflow.set_experiment("LangChain MLflow Integration")

mlflow.langchain.autolog()

"""
## Example: Tracing a LangChain Application

Here's a complete example showing MLflow tracing with LangChain:
"""
logger.info("## Example: Tracing a LangChain Application")


mlflow.langchain.autolog()

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

"""
To view the trace, run `mlflow ui` in your terminal and navigate to the Traces tab in the MLflow UI.

## Example: Tracing a LangGraph Application

MLflow also supports tracing LangGraph applications:
"""
logger.info("## Example: Tracing a LangGraph Application")


mlflow.langchain.autolog()


@tool
def count_words(text: str) -> str:
    """Counts the number of words in a text."""
    word_count = len(text.split())
    return f"This text contains {word_count} words."


llm = ChatOllama(model="llama3.2")
tools = [count_words]
graph = create_react_agent(llm, tools)

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Write me a 71-word story about a cat."}]}
)

"""
To view the trace, run `mlflow ui` in your terminal and navigate to the Traces tab in the MLflow UI.

## Resources

For more information on using MLflow with LangChain, please visit:

- [MLflow LangChain Integration Documentation](https://mlflow.org/docs/latest/llms/langchain/index.html)
- [MLflow Tracing Documentation](https://mlflow.org/docs/latest/llms/tracing/index.html)
- [Logging LangChain and LangGraph Models](https://mlflow.org/docs/latest/llms/langchain/index.html#logging-models-from-code)
- [Evaluating LangChain and LangGraph Models](https://mlflow.org/docs/latest/llms/langchain/index.html#how-can-i-evaluate-a-langgraph-agent)
"""
logger.info("## Resources")

logger.info("\n\n[DONE]", bright=True)