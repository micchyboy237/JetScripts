from enum import Enum
from io import BytesIO
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.adapters.ollama import convert_messages_for_finetuning
from langchain_community.chat_loaders.langsmith import LangSmithRunChatLoader
from langchain_core.output_parsers.ollama_functions import PydanticOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_pydantic_to_ollama_function
from langsmith.client import Client
from pprint import pprint
from pydantic import BaseModel
from pydantic import BaseModel, Field
import json
import ollama
import os
import shutil
import time
import uuid


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
# LangSmith LLM Runs

This notebook demonstrates how to directly load data from LangSmith's LLM runs and fine-tune a model on that data.
The process is simple and comprises 3 steps.

1. Select the LLM runs to train on.
2. Use the LangSmithRunChatLoader to load runs as chat sessions.
3. Fine-tune your model.

Then you can use the fine-tuned model in your LangChain app.

Before diving in, let's install our prerequisites.

## Prerequisites

Ensure you've installed langchain >= 0.0.311 and have configured your environment with your LangSmith API key.
"""
logger.info("# LangSmith LLM Runs")

# %pip install --upgrade --quiet  langchain langchain-ollama


uid = uuid.uuid4().hex[:6]
project_name = f"Run Fine-tuning Walkthrough {uid}"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "YOUR API KEY"
os.environ["LANGSMITH_PROJECT"] = project_name

"""
## 1. Select Runs
The first step is selecting which runs to fine-tune on. A common case would be to select LLM runs within
traces that have received positive user feedback. You can find examples of this in the[LangSmith Cookbook](https://github.com/langchain-ai/langsmith-cookbook/blob/main/exploratory-data-analysis/exporting-llm-runs-and-feedback/llm_run_etl.ipynb) and in the [docs](https://docs.smith.langchain.com/tracing/use-cases/export-runs/local).

For the sake of this tutorial, we will generate some runs for you to use here. Let's try fine-tuning a
simple function-calling chain.
"""
logger.info("## 1. Select Runs")




class Operation(Enum):
    add = "+"
    subtract = "-"
    multiply = "*"
    divide = "/"


class Calculator(BaseModel):
    """A calculator function"""

    num1: float
    num2: float
    operation: Operation = Field(..., description="+,-,*,/")

    def calculate(self):
        if self.operation == Operation.add:
            return self.num1 + self.num2
        elif self.operation == Operation.subtract:
            return self.num1 - self.num2
        elif self.operation == Operation.multiply:
            return self.num1 * self.num2
        elif self.operation == Operation.divide:
            if self.num2 != 0:
                return self.num1 / self.num2
            else:
                return "Cannot divide by zero"



ollama_function_def = convert_pydantic_to_ollama_function(Calculator)
plogger.debug(ollama_function_def)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an accounting assistant."),
        ("user", "{input}"),
    ]
)
chain = (
    prompt
    | ChatOllama(model="llama3.2").bind(functions=[ollama_function_def])
    | PydanticOutputFunctionsParser(pydantic_schema=Calculator)
    | (lambda x: x.calculate())
)

math_questions = [
    "What's 45/9?",
    "What's 81/9?",
    "What's 72/8?",
    "What's 56/7?",
    "What's 36/6?",
    "What's 64/8?",
    "What's 12*6?",
    "What's 8*8?",
    "What's 10*10?",
    "What's 11*11?",
    "What's 13*13?",
    "What's 45+30?",
    "What's 72+28?",
    "What's 56+44?",
    "What's 63+37?",
    "What's 70-35?",
    "What's 60-30?",
    "What's 50-25?",
    "What's 40-20?",
    "What's 30-15?",
]
results = chain.batch([{"input": q} for q in math_questions], return_exceptions=True)

"""
#### Load runs that did not error

Now we can select the successful runs to fine-tune on.
"""
logger.info("#### Load runs that did not error")


client = Client()

successful_traces = {
    run.trace_id
    for run in client.list_runs(
        project_name=project_name,
        execution_order=1,
        error=False,
    )
}

llm_runs = [
    run
    for run in client.list_runs(
        project_name=project_name,
        run_type="llm",
    )
    if run.trace_id in successful_traces
]

"""
## 2. Prepare data
Now we can create an instance of LangSmithRunChatLoader and load the chat sessions using its lazy_load() method.
"""
logger.info("## 2. Prepare data")


loader = LangSmithRunChatLoader(runs=llm_runs)

chat_sessions = loader.lazy_load()

"""
#### With the chat sessions loaded, convert them into a format suitable for fine-tuning.
"""
logger.info("#### With the chat sessions loaded, convert them into a format suitable for fine-tuning.")


training_data = convert_messages_for_finetuning(chat_sessions)

"""
## 3. Fine-tune the model
Now, initiate the fine-tuning process using the Ollama library.
"""
logger.info("## 3. Fine-tune the model")



my_file = BytesIO()
for dialog in training_data:
    my_file.write((json.dumps({"messages": dialog}) + "\n").encode("utf-8"))

my_file.seek(0)
training_file = ollama.files.create(file=my_file, purpose="fine-tune")

job = ollama.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="llama3.2",
)

status = ollama.fine_tuning.jobs.retrieve(job.id).status
start_time = time.time()
while status != "succeeded":
    logger.debug(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    status = ollama.fine_tuning.jobs.retrieve(job.id).status

"""
## 4. Use in LangChain

After fine-tuning, use the resulting model ID with the ChatOllama model class in your LangChain app.
"""
logger.info("## 4. Use in LangChain")

job = ollama.fine_tuning.jobs.retrieve(job.id)
model_id = job.fine_tuned_model


model = ChatOllama(
    model=model_id,
    temperature=1,
)

(prompt | model).invoke({"input": "What's 56/7?"})

"""
Now you have successfully fine-tuned a model using data from LangSmith LLM runs!
"""
logger.info("Now you have successfully fine-tuned a model using data from LangSmith LLM runs!")

logger.info("\n\n[DONE]", bright=True)