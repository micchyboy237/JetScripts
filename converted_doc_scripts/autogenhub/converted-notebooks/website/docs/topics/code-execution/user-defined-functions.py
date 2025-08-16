from autogen import ConversableAgent
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
from autogen.coding.func_with_reqs import Alias, ImportFromModule, with_requirements
from autogen.coding.func_with_reqs import with_requirements
from jet.logger import CustomLogger
from pandas import DataFrame
from pandas import DataFrame as df
from pathlib import Path
from {LocalCommandLineCodeExecutor.FUNCTIONS_MODULE} import add_two_numbers
from {LocalCommandLineCodeExecutor.FUNCTIONS_MODULE} import load_data
import os
import pandas
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# User Defined Functions

````{=mdx}
:::note
This is experimental and not *yet* supported by all executors. At this stage only [`LocalCommandLineCodeExecutor`](/docs/reference/coding/local_commandline_code_executor#localcommandlinecodeexecutor) is supported.


Currently, the method of registering tools and using this feature are different. We would like to unify them. See Github issue [here](https://github.com/microsoft/autogen/issues/2101)
:::
````

User defined functions allow you to define Python functions in your AutoGen program and then provide these to be used by your executor. This allows you to provide your agents with tools without using traditional tool calling APIs. Currently only Python is supported for this feature.

There are several steps involved:

1. Define the function
2. Provide the function to the executor
3. Explain to the code writing agent how to use the function


## Define the function

````{=mdx}
:::warning
Keep in mind that the entire source code of these functions will be available to the executor. This means that you should not include any sensitive information in the function as an LLM agent may be able to access it.
:::
````

If the function does not require any external imports or dependencies then you can simply use the function. For example:
"""
logger.info("# User Defined Functions")

def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

"""
This would be a valid standalone function.

````{=mdx}
:::tip
Using type hints and docstrings are not required but are highly recommended. They will help the code writing agent understand the function and how to use it.
:::
````

If the function requires external imports or dependencies then you can use the `@with_requirements` decorator to specify the requirements. For example:
"""
logger.info("This would be a valid standalone function.")




@with_requirements(python_packages=["pandas"], global_imports=["pandas"])
def load_data() -> pandas.DataFrame:
    """Load some sample data.

    Returns:
        pandas.DataFrame: A DataFrame with the following columns: name(str), location(str), age(int)
    """
    data = {
        "name": ["John", "Anna", "Peter", "Linda"],
        "location": ["New York", "Paris", "Berlin", "London"],
        "age": [24, 13, 53, 33],
    }
    return pandas.DataFrame(data)

"""
If you wanted to rename `pandas` to `pd` or import `DataFrame` directly you could do the following:
"""
logger.info("If you wanted to rename `pandas` to `pd` or import `DataFrame` directly you could do the following:")




@with_requirements(python_packages=["pandas"], global_imports=[Alias("pandas", "pd")])
def some_func1() -> pd.DataFrame: ...


@with_requirements(python_packages=["pandas"], global_imports=[ImportFromModule("pandas", "DataFrame")])
def some_func2() -> DataFrame: ...


@with_requirements(python_packages=["pandas"], global_imports=[ImportFromModule("pandas", Alias("DataFrame", "df"))])
def some_func3() -> df: ...

"""
## Provide the function to the executor

Functions can be loaded into the executor in its constructor. For example:
"""
logger.info("## Provide the function to the executor")



work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

executor = LocalCommandLineCodeExecutor(work_dir=work_dir, functions=[add_two_numbers, load_data])

"""
Before we get an agent involved, we can sanity check that when the agent writes code that looks like this the executor will be able to handle it.
"""
logger.info("Before we get an agent involved, we can sanity check that when the agent writes code that looks like this the executor will be able to handle it.")

code = f"""

logger.debug(add_two_numbers(1, 2))
"""

logger.debug(
    executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code=code),
        ]
    )
)

"""
And we can try the function that required a dependency and import too.
"""
logger.info("And we can try the function that required a dependency and import too.")

code = f"""

logger.debug(load_data())
"""

result = executor.execute_code_blocks(
    code_blocks=[
        CodeBlock(language="python", code=code),
    ]
)

logger.debug(result.output)

"""
### Limitations

- Only Python is supported currently
- The function must not depend on any globals or external state as it is loaded as source code 

## Explain to the code writing agent how to use the function

Now that the function is available to be called by the executor, you can explain to the code writing agent how to use the function. This step is very important as by default it will not know about it.

There is a utility function that you can use to generate a default prompt that describes the available functions and how to use them. This function can have its template overridden to provide a custom message, or you can use a different prompt all together.

For example, you could extend the system message from the page about local execution with a new section that describes the functions available.
"""
logger.info("### Limitations")

nlnl = "\n\n"
code_writer_system_message = """
You have been given coding capability to solve tasks using Python code.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
"""

code_writer_system_message += executor.format_functions_for_prompt()

logger.debug(code_writer_system_message)

"""
Then you can use this system message for your code writing agent.
"""
logger.info("Then you can use this system message for your code writing agent.")



code_writer_agent = ConversableAgent(
    "code_writer",
    system_message=code_writer_system_message,
#     llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
    code_execution_config=False,  # Turn off code execution for this agent.
    max_consecutive_auto_reply=2,
    human_input_mode="NEVER",
)

"""
Now, we can setup the code execution agent using the local command line executor we defined earlier.
"""
logger.info("Now, we can setup the code execution agent using the local command line executor we defined earlier.")

code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={
        "executor": executor,
    },
    human_input_mode="NEVER",
)

"""
Then, we can start the conversation and get the agent to process the dataframe we provided.
"""
logger.info("Then, we can start the conversation and get the agent to process the dataframe we provided.")

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message="Please use the load_data function to load the data and please calculate the average age of all people.",
    summary_method="reflection_with_llm",
)

"""
We can see the summary of the calculation:
"""
logger.info("We can see the summary of the calculation:")

logger.debug(chat_result.summary)

logger.info("\n\n[DONE]", bright=True)