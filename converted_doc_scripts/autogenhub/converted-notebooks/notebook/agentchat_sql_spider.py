from autogen import ConversableAgent, UserProxyAgent, config_list_from_json
from jet.logger import CustomLogger
from spider_env import SpiderEnv
from typing import Annotated, Dict
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# SQL Agent for Spider text-to-SQL benchmark

This notebook demonstrates a basic SQL agent that translates natural language questions into SQL queries.

## Environment

For this demo, we use a SQLite database environment based on a standard text-to-sql benchmark called [Spider](https://yale-lily.github.io/spider). The environment provides a gym-like interface and can be used as follows.
"""
logger.info("# SQL Agent for Spider text-to-SQL benchmark")




gym = SpiderEnv()

observation, info = gym.reset()

question = observation["instruction"]
logger.debug(question)

schema = info["schema"]
logger.debug(schema)

"""
## Agent Implementation

Using AutoGen, a SQL agent can be implemented with a ConversableAgent. The gym environment executes the generated SQL query and the agent can take execution results as feedback to improve its generation in multiple rounds of conversations.
"""
logger.info("## Agent Implementation")

os.environ["AUTOGEN_USE_DOCKER"] = "False"
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")


def check_termination(msg: Dict):
    if "tool_responses" not in msg:
        return False
    json_str = msg["tool_responses"][0]["content"]
    obj = json.loads(json_str)
    return "error" not in obj or obj["error"] is None and obj["reward"] == 1


sql_writer = ConversableAgent(
    "sql_writer",
    llm_config={"config_list": config_list},
    system_message="You are good at writing SQL queries. Always respond with a function call to execute_sql().",
    is_termination_msg=check_termination,
)
user_proxy = UserProxyAgent("user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=5)


@sql_writer.register_for_llm(description="Function for executing SQL query and returning a response")
@user_proxy.register_for_execution()
def execute_sql(
    reflection: Annotated[str, "Think about what to do"], sql: Annotated[str, "SQL query"]
) -> Annotated[Dict[str, str], "Dictionary with keys 'result' and 'error'"]:
    observation, reward, _, _, info = gym.step(sql)
    error = observation["feedback"]["error"]
    if not error and reward == 0:
        error = "The SQL query returned an incorrect result"
    if error:
        return {
            "error": error,
            "wrong_result": observation["feedback"]["result"],
            "correct_result": info["gold_result"],
        }
    else:
        return {
            "result": observation["feedback"]["result"],
        }

"""
The agent can then take as input the schema and the text question, and generate the SQL query.
"""
logger.info("The agent can then take as input the schema and the text question, and generate the SQL query.")

message = f"""Below is the schema for a SQL database:
{schema}
Generate a SQL query to answer the following question:
{question}
"""

user_proxy.initiate_chat(sql_writer, message=message)

logger.info("\n\n[DONE]", bright=True)