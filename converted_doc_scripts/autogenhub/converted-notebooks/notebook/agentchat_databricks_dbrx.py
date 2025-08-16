from autogen import AssistantAgent, UserProxyAgent
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
from jet.logger import CustomLogger
from pathlib import Path
from pyspark.sql import SparkSession
import autogen
import autogen.runtime_logging
import os
import pandas as pd
import shutil
import sqlite3


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Use AutoGen in Databricks with DBRX

![DBRX launch](https://www.databricks.com/en-blog-assets/static/2fe1a0af1ee0f6605024a810b604079c/dbrx-blog-header-optimized.png)

In March 2024, Databricks released [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), a general-purpose LLM that sets a new standard for open LLMs. While available as an open-source model on Hugging Face ([databricks/dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct/tree/main) and [databricks/dbrx-base](https://huggingface.co/databricks/dbrx-base) ), customers of Databricks can also tap into the [Foundation Model APIs](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#query-a-chat-completion-model), which make DBRX available through an Ollama-compatible, autoscaling REST API.

[Autogen](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat) is becoming a popular standard for agent creation. Built to support any "LLM as a service" that implements the Ollama SDK, it can easily be extended to integrate with powerful open source models. 

This notebook will demonstrate a few basic examples of Autogen with DBRX, including the use of  `AssistantAgent`, `UserProxyAgent`, and `ConversableAgent`. These demos are not intended to be exhaustive - feel free to use them as a base to build upon!

## Requirements
AutoGen must be installed on your Databricks cluster, and requires `Python>=3.8`. This example includes the `%pip` magic command to install: `%pip install pyautogen`, as well as other necessary libraries. 

This code has been tested on: 
* [Serverless Notebooks](https://docs.databricks.com/en/compute/serverless.html) (in public preview as of Apr 18, 2024)
* Databricks Runtime 14.3 LTS ML [docs](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html)

This code can run in any Databricks workspace in a region where DBRX is available via pay-per-token APIs (or provisioned throughput). To check if your region is supported, see [Foundation Model Region Availability](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-limits.html#foundation-model-apis-limits). If the above is true, the workspace must also be enabled by an admin for Foundation Model APIs [docs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html#requirements).

## Tips
* This notebook can be imported from github to a Databricks workspace and run directly. Use [sparse checkout mode with git](https://www.databricks.com/blog/2023/01/26/work-large-monorepos-sparse-checkout-support-databricks-repos.html) to import only this notebook or the examples directory. 

* Databricks recommends using [Secrets](https://docs.databricks.com/en/security/secrets/secrets.html) instead of storing tokens in plain text. 

## Contributor

tj@databricks.com (Github: tj-cycyota)
"""
logger.info("# Use AutoGen in Databricks with DBRX")

# %pip install pyautogen==0.2.25 openai==1.21.2 typing_extensions==4.11.0 --upgrade

"""
It is recommended to restart the Python kernel after installs - uncomment and run the below:
"""
logger.info("It is recommended to restart the Python kernel after installs - uncomment and run the below:")



"""
## Setup DBRX config list

See Autogen docs for more inforation on the use of `config_list`: [LLM Configuration](https://autogenhub.github.io/autogen/docs/topics/llm_configuration#why-is-it-a-list)
"""
logger.info("## Setup DBRX config list")



os.environ["DATABRICKS_HOST"] = "<FILL IN WITH YOUR WORKSPACE URL IN SUPPORTED DBRX REGION>"

os.environ["DATABRICKS_TOKEN"] = "dapi...."

llm_config = {
    "config_list": [
        {
            "model": "databricks-dbrx-instruct",
            "api_key": str(os.environ["DATABRICKS_TOKEN"]),
            "base_url": str(os.getenv("DATABRICKS_HOST")) + "/serving-endpoints",
        }
    ],
}

"""
## Hello World Example

Our first example will be with a simple `UserProxyAgent` asking a question to an `AssistantAgent`. This is based on the tutorial demo [here](https://autogenhub.github.io/autogen/docs/tutorial/introduction).

After sending the question and seeing a response, you can type `exit` to end the chat or continue to converse.
"""
logger.info("## Hello World Example")


assistant = autogen.AssistantAgent(name="assistant", llm_config=llm_config)

user_proxy = autogen.UserProxyAgent(name="user", code_execution_config=False)

chat_result = user_proxy.initiate_chat(assistant, message="What is MLflow?")

"""
## Simple Coding Agent

In this example, we will implement a "coding agent" that can execute code. You will see how this code is run alongside your notebook in your current workspace, taking advantage of the performance benefits of Databricks clusters. This is based off the demo [here](https://autogenhub.github.io/autogen/docs/topics/non-openai-models/cloud-mistralai/).

First, set up a directory:
"""
logger.info("## Simple Coding Agent")


workdir = Path("coding")
logger.debug(workdir)
workdir.mkdir(exist_ok=True)


code_executor = LocalCommandLineCodeExecutor(work_dir=workdir)

"""
Next, setup our agents and initiate a coding problem. Notice how the `UserProxyAgent` will take advantage of our `code_executor`; after the code is shown on screen, type Return/Enter in the chatbox to have it execute locally on your cluster via the bot's auto-reply. 

**Note**: with generative AI coding assistants, you should **always** manually read and review the code before executing it yourself, as LLM results are non-deterministic and may lead to unintended consequences.
"""
logger.info("Next, setup our agents and initiate a coding problem. Notice how the `UserProxyAgent` will take advantage of our `code_executor`; after the code is shown on screen, type Return/Enter in the chatbox to have it execute locally on your cluster via the bot's auto-reply.")


user_proxy_agent = UserProxyAgent(
    name="User",
    code_execution_config={"executor": code_executor},
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content"),
)

assistant_agent = AssistantAgent(
    name="DBRX Assistant",
    llm_config=llm_config,
)

chat_result = user_proxy_agent.initiate_chat(
    assistant_agent,
    message="Count how many prime numbers from 1 to 10000.",
)

"""
We can see the python file that was created in our working directory:
"""
logger.info("We can see the python file that was created in our working directory:")

# %sh ls coding

# %sh head coding/count_primes.py

"""
## Conversable Bots

We can also implement the [two-agent chat pattern](https://autogenhub.github.io/autogen/docs/tutorial/conversation-patterns/#two-agent-chat-and-chat-result) using DBRX to "talk to itself" in a teacher/student exchange:
"""
logger.info("## Conversable Bots")


student_agent = ConversableAgent(
    name="Student_Agent",
    system_message="You are a student willing to learn.",
    llm_config=llm_config,
)

teacher_agent = ConversableAgent(
    name="Teacher_Agent",
    system_message="You are a computer science teacher.",
    llm_config=llm_config,
)

chat_result = student_agent.initiate_chat(
    teacher_agent,
    message="How does deep learning relate to artificial intelligence?",
    summary_method="last_msg",
    max_turns=1,  # Set to higher number to control back and forth
)

"""
## Implement Logging Display

It can be useful to display chat logs to the notebook for debugging, and then persist those logs to a Delta table. The following section demonstrates how to extend the default AutoGen logging libraries.

First, we will implement a Python `class` that extends the capabilities of `autogen.runtime_logging` [docs](https://autogenhub.github.io/autogen/docs/notebooks/agentchat_logging):
"""
logger.info("## Implement Logging Display")

class Databricks_AutoGenLogger:
    def __init__(self):

        self.spark = SparkSession.builder.getOrCreate()
        self.logger_config = {"dbname": "logs.db"}

    def start(self):

        self.logging_session_id = autogen.runtime_logging.start(config=self.logger_config)
        logger.debug("Logging session ID: " + str(self.logging_session_id))

    def stop(self):

        autogen.runtime_logging.stop()

    def _get_log(self, dbname="logs.db", table="chat_completions"):

        con = sqlite3.connect(dbname)
        query = f"SELECT * from {table} WHERE session_id == '{self.logging_session_id}' ORDER BY end_time DESC"
        cursor = con.execute(query)
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        data = [dict(zip(column_names, row)) for row in rows]
        con.close()
        return data

    def display_session(self):

        return pd.DataFrame(self._get_log())

    def persist_results(self, target_delta_table: str, mode="append"):

        sdf = self.spark.createDataFrame(pd.DataFrame(self._get_log()))

        try:
            sdf.write.format("delta").mode(mode).saveAsTable(target_delta_table)
            logger.debug(f"Logs sucessfully written to table {target_delta_table} in {mode} mode")
        except Exception as e:
            logger.debug(f"An error occurred: {e}")

"""
Let's use the class above on our simplest example. Note the addition of logging `.start()` and `.stop()`, as well as try/except for error handling.
"""
logger.info("Let's use the class above on our simplest example. Note the addition of logging `.start()` and `.stop()`, as well as try/except for error handling.")

assistant = autogen.AssistantAgent(name="assistant", llm_config=llm_config)
user_proxy = autogen.UserProxyAgent(name="user", code_execution_config=False)

logs = Databricks_AutoGenLogger()
logs.start()
try:
    user_proxy.initiate_chat(assistant, message="What is MLflow?", max_turns=1)
except Exception as e:
    logger.debug(f"An error occurred: {e}")
logs.stop()
display(logs.display_session())

"""
With this, we have a simple framework to review and persist logs from our chats! Notice that in the `request` field above, we can also see the system prompt for the LLM - this can be useful for prompt engineering as well as debugging.

Note that when you deploy this to Databricks Model Serving, model responses are auto-logged using [Lakehouse Monitoring](https://docs.databricks.com/en/lakehouse-monitoring/index.html); but the above approach provides a simple mechanism to log chats from the **client side**.

Let's now persist these results to a Delta table in [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html):
"""
logger.info("With this, we have a simple framework to review and persist logs from our chats! Notice that in the `request` field above, we can also see the system prompt for the LLM - this can be useful for prompt engineering as well as debugging.")


spark = SparkSession.builder.getOrCreate()  # Not needed in Databricks; session pre-provisioned in notebooks

target_delta_table = "your_catalog.your_schema.autogen_logs"
logs.persist_results(target_delta_table=target_delta_table, mode="append")

display(spark.table(target_delta_table))

"""
## Closing Thoughts
This notebook provides a few basic examples of using Autogen with DBRX, and we're excited to see how you can use this framework alongside leading open-source LLMs!

### Limitations
* Databricks Foundation Model API supports other open-source LLMs (Mixtral, Llama2, etc.), but the above code has not been tested on those.

* As of April 2024, DBRX does not yet support tool/function calling abilities. To discuss this capability further, please reach out to your Databricks representative.
"""
logger.info("## Closing Thoughts")

logger.info("\n\n[DONE]", bright=True)