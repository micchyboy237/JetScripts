from autogen.code_utils import extract_code
from eventlet.timeout import Timeout
from gurobipy import GRB
from jet.logger import CustomLogger
from termcolor import colored
from typing import Union
import autogen
import os
import re
import requests  # for loading the example source code
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# OptiGuide with Nested Chats in AutoGen 

This is a nested chat re-implementation of [OptiGuide](https://github.com/microsoft/OptiGuide), which is an LLM-based supply chain optimization framework. 
 
 
In addition to AutoGen, this notebook also requires eventlet and Gurobipy.  The eventlet package is used in this notebook to constrain code execution with a timeout, and the gurobipy package is a mathematical optimization software library for solving mixed-integer linear and quadratic optimization problems. 

````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install pyautogen eventlet gurobipy
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````
"""
logger.info("# OptiGuide with Nested Chats in AutoGen")





config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt4", "gpt-3.5-turbo" "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)
llm_config = {"config_list": config_list_gpt4}

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

Intended agent orchestration in OptiGuide.

![](optiGuide_new_design.png)



## Step 0. Prepare Helper Functions
The code cell below includes several helper functions to be used by agents. The helper functions are adopted directly from [OptiGuide](https://github.com/microsoft/OptiGuide).

### Utility Functions
Several utility functions (replace, insert_code, run_with_exec) are defined to manipulate and execute the source code dynamically:

- **replace(src_code, old_code, new_code) -> str:**<br/>Replaces a specified block of code within the source code with new code. This is essential for updating the code dynamically based on chatbot and user interactions.

- **insert_code(src_code, new_lines) -> str:**<br/>Determines where to insert new lines of code based on specific markers in the source code. It's used to seamlessly integrate generated code snippets.

- **run_with_exec(src_code) -> Union[str, Exception]:**<br/>Executes the modified source code within a controlled environment, capturing and returning the output or any errors that occur. This function is crucial for testing the feasibility and correctness of the generated code solutions.

### Functions External Code Retrieval
The cell also includes logic to retrieve example source code from a remote repository using the requests library. This example code serves as a base for the chatbot to work with, allowing users to see real-life applications of the concepts discussed.

The retrieved code is then displayed, showing both the beginning and the end of the source code to give a glimpse of its structure and content.
"""
logger.info("## Step 0. Prepare Helper Functions")

def replace(src_code: str, old_code: str, new_code: str) -> str:
    """
    Inserts new code into the source code by replacing a specified old
    code block.

    Args:
        src_code (str): The source code to modify.
        old_code (str): The code block to be replaced.
        new_code (str): The new code block to insert.

    Returns:
        str: The modified source code with the new code inserted.

    Raises:
        None

    Example:
        src_code = 'def hello_world():\n    logger.debug("Hello, world!")\n\n# Some
        other code here'
        old_code = 'logger.debug("Hello, world!")'
        new_code = 'logger.debug("Bonjour, monde!")\nlogger.debug("Hola, mundo!")'
        modified_code = _replace(src_code, old_code, new_code)
        logger.debug(modified_code)
    """
    pattern = r"( *){old_code}".format(old_code=old_code)
    head_spaces = re.search(pattern, src_code, flags=re.DOTALL).group(1)
    new_code = "\n".join([head_spaces + line for line in new_code.split("\n")])
    rst = re.sub(pattern, new_code, src_code)
    return rst


def insert_code(src_code: str, new_lines: str) -> str:
    """insert a code patch into the source code.


    Args:
        src_code (str): the full source code
        new_lines (str): The new code.

    Returns:
        str: the full source code after insertion (replacement).
    """
    if new_lines.find("addConstr") >= 0:
        return replace(src_code, CONSTRAINT_CODE_STR, new_lines)
    else:
        return replace(src_code, DATA_CODE_STR, new_lines)


def run_with_exec(src_code: str) -> Union[str, Exception]:
    """Run the code snippet with exec.

    Args:
        src_code (str): The source code to run.

    Returns:
        object: The result of the code snippet.
            If the code succeed, returns the objective value (float or string).
            else, return the error (exception)
    """
    locals_dict = {}
    locals_dict.update(globals())
    locals_dict.update(locals())

    timeout = Timeout(
        60,
        TimeoutError("This is a timeout exception, in case " "GPT's code falls into infinite loop."),
    )
    try:
        exec(src_code, locals_dict, locals_dict)
    except Exception as e:
        return e
    finally:
        timeout.cancel()

    try:
        status = locals_dict["m"].Status
        if status != GRB.OPTIMAL:
            if status == GRB.UNBOUNDED:
                ans = "unbounded"
            elif status == GRB.INF_OR_UNBD:
                ans = "inf_or_unbound"
            elif status == GRB.INFEASIBLE:
                ans = "infeasible"
                m = locals_dict["m"]
                m.computeIIS()
                constrs = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
                ans += "\nConflicting Constraints:\n" + str(constrs)
            else:
                ans = "Model Status:" + str(status)
        else:
            ans = "Optimization problem solved. The objective value is: " + str(locals_dict["m"].objVal)
    except Exception as e:
        return e

    return ans

"""
## Step 1. Agent Construction

This cell introduces the Writer and OptiGuide agent classes and their instances to manage the interaction between the user, the chatbot, and the optimization solver. This streamlines the process of generating, evaluating, and integrating code solutions for supply chain optimization problems.

### Classes Defined

- **`OptiGuide`**: Inherits from `autogen.AssistantAgent` and serves as the main class for handling the supply chain optimization logic. It maintains state information like the source code, debugging attempts left, success status, and user chat history. Key methods include `set_success` and `update_debug_times`, which are used to update the agent's state based on the outcomes of interactions.

- **`Writer`**: Also inherits from `autogen.AssistantAgent`, this class is tailored to manage the generation and explanation of Python code solutions. It keeps track of the source code and example Q&A to assist in generating responses to user queries.

### Agent Instances

- **`writer`**, **`safeguard`**, and **`optiguide_commander`**: Instances of the classes defined above, each configured with specific roles in the code generation and evaluation process. These agents work together to ensure that the user's questions are answered with safe, optimized, and understandable Python code solutions.

- **`user`**: An instance of `autogen.UserProxyAgent`, representing the end-user interacting with the notebook. It's configured to simulate user inputs and responses, facilitating an interactive and engaging learning experience.

#### Agent System Messages
Each agent is associted with a corresponding system message. Those system messages are adopted directly from [OptiGuide](https://github.com/microsoft/OptiGuide).
"""
logger.info("## Step 1. Agent Construction")

WRITER_SYSTEM_MSG = """You are a chatbot to:
(1) write Python code to answer users questions for supply chain-related coding
project;
(2) explain solutions from a Gurobi/Python solver.

--- SOURCE CODE ---
{source_code}

--- DOC STR ---
{doc_str}
---

Here are some example questions and their answers and codes:
--- EXAMPLES ---
{example_qa}
---

The execution result of the original source code is below.
--- Original Result ---
{execution_result}

Note that your written code will be added to the lines with substring:
"# OPTIGUIDE *** CODE GOES HERE"
So, you don't need to write other code, such as m.optimize() or m.update().
You just need to write code snippet in ```python ...``` block.
"""

SAFEGUARD_SYSTEM_MSG = """
Given the source code:
{source_code}

Is the source code safe (not malicious code to break security
and privacy) to run?
Answer only one word.
If not safe, answer `DANGER`; else, answer `SAFE`.
"""

DATA_CODE_STR = "# OPTIGUIDE DATA CODE GOES HERE"
CONSTRAINT_CODE_STR = "# OPTIGUIDE CONSTRAINT CODE GOES HERE"

example_qa = """
----------
Question: Why is it not recommended to use just one supplier for roastery 2?
Answer Code:
```python
z = m.addVars(suppliers, vtype=GRB.BINARY, name="z")
m.addConstr(sum(z[s] for s in suppliers) <= 1, "_")
for s in suppliers:
    m.addConstr(x[s,'roastery2'] <= capacity_in_supplier[s] * z[s], "_")
```

----------
Question: What if there's a 13% jump in the demand for light coffee at cafe1?
Answer Code:
```python
light_coffee_needed_for_cafe["cafe1"] = light_coffee_needed_for_cafe["cafe1"] * (1 + 13/100)
```

"""

CODE_PROMPT = """
Answer Code:
"""

DEBUG_PROMPT = """

While running the code you suggested, I encountered the {error_type}:
--- ERROR MESSAGE ---
{error_message}

Please try to resolve this bug, and rewrite the code snippet.
--- NEW CODE ---
"""

SAFEGUARD_PROMPT = """
--- Code ---
{code}

--- One-Word Answer: SAFE or DANGER ---
"""

INTERPRETER_PROMPT = """Here are the execution results: {execution_rst}

Can you organize these information to a human readable answer?
Remember to compare the new results to the original results you obtained in the
beginning.

--- HUMAN READABLE ANSWER ---
"""

code_url = "https://raw.githubusercontent.com/microsoft/OptiGuide/main/benchmark/application/coffee.py"
response = requests.get(code_url)
if response.status_code == 200:
    code = response.text
else:
    raise RuntimeError("Failed to retrieve the file.")


logger.debug("\n".join(code.split("\n")[:10]))
logger.debug(".\n" * 3)
logger.debug("\n".join(code.split("\n")[-10:]))

writer_system_msg = WRITER_SYSTEM_MSG.format(
    source_code=code,
    doc_str="",
    example_qa=example_qa,
    execution_result="",
)
safeguard_system_msg = SAFEGUARD_SYSTEM_MSG.format(source_code=code)


class OptiGuide(autogen.AssistantAgent):
    source_code: str = code
    debug_times: int = 3
    debug_times_left: int = 3
    example_qa: str = ""
    success: bool = False
    user_chat_history: str = ""


class Writer(autogen.AssistantAgent):
    source_code: str = code
    example_qa: str = ""
    user_chat_history: str = ""


writer = Writer("writer", llm_config=llm_config)
safeguard = autogen.AssistantAgent("safeguard", llm_config=llm_config)
optiguide_commander = OptiGuide("commander", llm_config=llm_config)

user = autogen.UserProxyAgent(
    "user", max_consecutive_auto_reply=0, human_input_mode="NEVER", code_execution_config=False
)

"""
## Step 2. Orchestrate Nested Chats 

These three agent instances are orchestrated in the following way with the `writer` and `safeguard` nested into the `commander` agent as the inner monologue.

This next cell defines critical functions that manage the interactions between the `OptiGuide` system, the user, and the internal logic for processing and responding to queries. Each function plays a specific role in guiding the conversation, handling code generation requests, ensuring code safety, and summarizing outcomes.  This ensures that agents receive clear instructions, immediate feedback, and a secure environment for exploring supply chain optimization problems through code.

Information about the sequence of chats can be specified in the `chat_queue` argument of the `register_nested_chats` function. The following fields are especially useful:
- `recipient` (required) specifies the nested agent;
- `message` specifies what message to send to the nested recipient agent. In a sequence of nested chats, if the `message` field is not specified, we will use the last message the registering agent received as the initial message in the first chat and will skip any subsequent chat in the queue that does not have the `message` field. You can either provide a string or define a callable that returns a string.
- `summary_method` decides what to get out of the nested chat. You can either select from existing options including "last_msg" and "reflection_with_llm", or or define your own way on what to get from the nested chat with a Callable.
- `max_turns` determines how many turns of conversation to have between the concerned agent pairs.
"""
logger.info("## Step 2. Orchestrate Nested Chats")

def writer_init_messsage(recipient, messages, sender, config):
    if recipient.success:
        return None
    msg_content = messages[-1].get("content", "")
    sender_history = recipient.chat_messages[sender]
    user_chat_history = "\nHere are the history of discussions:\n" f"{sender_history}"

    if sender.name == "user":
        execution_result = msg_content  # TODO: get the execution result of the original source code
    else:
        execution_result = ""
    writer_sys_msg = (
        WRITER_SYSTEM_MSG.format(
            source_code=recipient.source_code,
            doc_str="",
            example_qa=example_qa,
            execution_result=execution_result,
        )
        + user_chat_history
    )

    recipient.debug_times_left = recipient.debug_times
    recipient.success = False
    return writer_sys_msg + "\n" + CODE_PROMPT


def writer_success_summary(recipient, sender):
    if sender.success:
        return sender.last_message(recipient)["content"].replace("TERMINATE", "")
    else:
        return "Sorry. I cannot answer your question."


def safeguard_init_message(recipient, messages, sender, config):
    if recipient.success:
        return None
    last_msg_content = messages[-1].get("content", "")
    _, code = extract_code(last_msg_content)[0]
    if _ != "unknown":
        return SAFEGUARD_SYSTEM_MSG.format(source_code=code) + sender.user_chat_history
    else:
        return


def safeguard_summary(recipient, sender):
    safe_msg = sender.last_message(recipient)["content"].replace("TERMINATE", "")

    if safe_msg.find("DANGER") < 0:
        src_code = insert_code(sender.source_code, code)
        execution_rst = run_with_exec(src_code)
        logger.debug(colored(str(execution_rst), "yellow"))
        if type(execution_rst) in [str, int, float]:
            sender.success = True
            return INTERPRETER_PROMPT.format(execution_rst=execution_rst)
    else:
        execution_rst = """
        Sorry, this new code is not safe to run. I would not allow you to execute it.
        Please try to find a new way (coding) to answer the question."""
        if sender.debug_times_left > 0:
            sender.debug_times_left -= 1
            return DEBUG_PROMPT.format(error_type=type(execution_rst), error_message=str(execution_rst))


writer_chat_queue = [{"recipient": writer, "message": writer_init_messsage, "summary_method": writer_success_summary}]
safeguard_chat_queue = [
    {"recipient": safeguard, "message": safeguard_init_message, "max_turns": 1, "summary_method": safeguard_summary}
]
optiguide_commander.register_nested_chats(safeguard_chat_queue, trigger="writer")
optiguide_commander.register_nested_chats(writer_chat_queue, trigger="user")

"""
### Let the agents talk
"""
logger.info("### Let the agents talk")

chat_res = user.initiate_chat(
    optiguide_commander, message="What if we prohibit shipping from supplier 1 to roastery 2?"
)

"""
### Get Final Results from the Returned ChatResult Object
"""
logger.info("### Get Final Results from the Returned ChatResult Object")

logger.debug(chat_res.summary)

logger.info("\n\n[DONE]", bright=True)