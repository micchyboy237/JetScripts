from autogen import AssistantAgent, UserProxyAgent
from jet.logger import CustomLogger
import autogen
import json
import os
import pandas as pd
import sqlite3

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Runtime Logging with AutoGen 

AutoGen offers utilities to log data for debugging and performance analysis. This notebook demonstrates how to use them. 

we log data in different modes:
- SQlite Database
- File 

In general, users can initiate logging by calling `autogen.runtime_logging.start()` and stop logging by calling `autogen.runtime_logging.stop()`
"""
logger.info("# Runtime Logging with AutoGen")




llm_config = {
    "config_list": autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
    ),
    "temperature": 0.9,
}

logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
logger.debug("Logging session ID: " + str(logging_session_id))

assistant = AssistantAgent(name="assistant", llm_config=llm_config)
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)

user_proxy.initiate_chat(
    assistant, message="What is the height of the Eiffel Tower? Only respond with the answer and terminate"
)
autogen.runtime_logging.stop()

"""
## Getting Data from the SQLite Database 

`logs.db` should be generated, by default it's using SQLite database. You can view the data with GUI tool like `sqlitebrowser`, using SQLite command line shell or using python script:
"""
logger.info("## Getting Data from the SQLite Database")

def get_log(dbname="logs.db", table="chat_completions"):

    con = sqlite3.connect(dbname)
    query = f"SELECT * from {table}"
    cursor = con.execute(query)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    data = [dict(zip(column_names, row)) for row in rows]
    con.close()
    return data

def str_to_dict(s):
    return json.loads(s)


log_data = get_log()
log_data_df = pd.DataFrame(log_data)

log_data_df["total_tokens"] = log_data_df.apply(
    lambda row: str_to_dict(row["response"])["usage"]["total_tokens"], axis=1
)

log_data_df["request"] = log_data_df.apply(lambda row: str_to_dict(row["request"])["messages"][0]["content"], axis=1)

log_data_df["response"] = log_data_df.apply(
    lambda row: str_to_dict(row["response"])["choices"][0]["message"]["content"], axis=1
)

log_data_df

"""
## Computing Cost 

One use case of logging data is to compute the cost of a session.
"""
logger.info("## Computing Cost")

total_tokens = log_data_df["total_tokens"].sum()

total_cost = log_data_df["cost"].sum()

session_tokens = log_data_df[log_data_df["session_id"] == logging_session_id]["total_tokens"].sum()
session_cost = log_data_df[log_data_df["session_id"] == logging_session_id]["cost"].sum()

logger.debug("Total tokens for all sessions: " + str(total_tokens) + ", total cost: " + str(round(total_cost, 4)))
logger.debug(
    "Total tokens for session "
    + str(logging_session_id)
    + ": "
    + str(session_tokens)
    + ", cost: "
    + str(round(session_cost, 4))
)

"""
## Log data in File mode

By default, the log type is set to `sqlite` as shown above, but we introduced a new parameter for the `autogen.runtime_logging.start()`

the `logger_type = "file"` will start to log data in the File mode.
"""
logger.info("## Log data in File mode")



llm_config = {
    "config_list": autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
    ),
    "temperature": 0.9,
}

logging_session_id = autogen.runtime_logging.start(logger_type="file", config={"filename": "runtime.log"})
logger.debug("Logging session ID: " + str(logging_session_id))

assistant = AssistantAgent(name="assistant", llm_config=llm_config)
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)

user_proxy.initiate_chat(
    assistant, message="What is the height of the Eiffel Tower? Only respond with the answer and terminate"
)
autogen.runtime_logging.stop()

"""
This should create a `runtime.log` file in your current directory.
"""
logger.info("This should create a `runtime.log` file in your current directory.")

logger.info("\n\n[DONE]", bright=True)