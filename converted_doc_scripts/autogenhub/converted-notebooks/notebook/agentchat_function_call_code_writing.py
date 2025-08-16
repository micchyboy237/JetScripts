from fastapi import FastAPI
from spread_calculation import calculate_daily_spread
from typing_extensions import Annotated
import autogen
import os
import yfinance as yf

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Writing a software application using function calls

The default way of creating code in Autogen is its built-in code extractor. Although it allows for creating and executing simple scripts fast, that way of creating code is not suitable for developing advanced software applications, according to my experiences. The process of developing an application is mostly the process of introducing changes into existing files rather than creating new files with code. And in my experience, the code extractor is bad at introducing changes as the model often gets lost and can damage existing files.

Properly created functions that can modify code provide us with the ability to have more control over code changes and result in better quality. Additionally, as the scope of possible operations is predefined inside the tools, we can safely use Autogen without Docker, avoiding all the complications related to it.

## Requirements
"""
logger.info("# Writing a software application using function calls")

# ! pip install pyautogen

"""
## Set your API Endpoint
"""
logger.info("## Set your API Endpoint")



# config_list = [{"model": "gpt-4-turbo-preview", "api_key": os.getenv("OPENAI_API_KEY")}]

"""
## Create agents

In this example, we will improve a simple FastAPI application using only dedicated function calls. Let's create an Engineer agent that will think out and execute code changes, and a user proxy Admin agent, through which we will guide our Engineer.
"""
logger.info("## Create agents")

llm_config = {
    "temperature": 0,
    "config_list": config_list,
}

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    system_message="""
    I'm Engineer. I'm expert in python programming. I'm executing code tasks required by Admin.
    """,
)

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    human_input_mode="ALWAYS",
    code_execution_config=False,
)

"""
Mention, unlike in many other examples, here we don't need a separate Executor agent to save our code, as that will be done by functions. We also don't need Docker to be running because of that - which makes the entire process easier.

Next, let's set up our group chat.
"""
logger.info("Mention, unlike in many other examples, here we don't need a separate Executor agent to save our code, as that will be done by functions. We also don't need Docker to be running because of that - which makes the entire process easier.")

groupchat = autogen.GroupChat(
    agents=[engineer, user_proxy],
    messages=[],
    max_round=500,
    speaker_selection_method="round_robin",
    enable_clear_history=True,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

"""
## Prepare appropriate functions

Let's go to core of the thing. Prepare functions that provide Engineer with functionality to modify existing code, create new code files, check filesystem and files.
"""
logger.info("## Prepare appropriate functions")


default_path = "backend_dir/"


@user_proxy.register_for_execution()
@engineer.register_for_llm(description="List files in choosen directory.")
def list_dir(directory: Annotated[str, "Directory to check."]):
    files = os.listdir(default_path + directory)
    return 0, files


@user_proxy.register_for_execution()
@engineer.register_for_llm(description="Check the contents of a chosen file.")
def see_file(filename: Annotated[str, "Name and path of file to check."]):
    with open(default_path + filename, "r") as file:
        lines = file.readlines()
    formatted_lines = [f"{i+1}:{line}" for i, line in enumerate(lines)]
    file_contents = "".join(formatted_lines)

    return 0, file_contents


@user_proxy.register_for_execution()
@engineer.register_for_llm(description="Replace old piece of code with new one. Proper indentation is important.")
def modify_code(
    filename: Annotated[str, "Name and path of file to change."],
    start_line: Annotated[int, "Start line number to replace with new code."],
    end_line: Annotated[int, "End line number to replace with new code."],
    new_code: Annotated[str, "New piece of code to replace old code with. Remember about providing indents."],
):
    with open(default_path + filename, "r+") as file:
        file_contents = file.readlines()
        file_contents[start_line - 1 : end_line] = [new_code + "\n"]
        file.seek(0)
        file.truncate()
        file.write("".join(file_contents))
    return 0, "Code modified"


@user_proxy.register_for_execution()
@engineer.register_for_llm(description="Create a new file with code.")
def create_file_with_code(
    filename: Annotated[str, "Name and path of file to create."], code: Annotated[str, "Code to write in the file."]
):
    with open(default_path + filename, "w") as file:
        file.write(code)
    return 0, "File created successfully"

"""
## Prepare code to work with

In this example, we will show how AI can extend the functionalities of existing code, as improving existing code is a much more frequent use case in software development than creating a new one. Let's prepare the initial code on which we will work. That will be a simple FastAPI script that will allow you to calculate today's stock spread in percentage for CD Project Red, a famous Polish gamedev company. Create a folder called 'backend_dir' and place a 'main.py' file here with the following content:

```python
# backend_dir/main.py


app = FastAPI()

@app.get("/cdr_daily_spread")
async def calculate_daily_spread():
    cdr = yf.Ticker("CDR.WA")
    today_data = cdr.history(period="1d")
    spread = ((today_data["High"] - today_data["Low"]) / today_data["Low"]) * 100
    return spread
```

Install needed libraries. We can run our API using:

```bash
uvicorn main:app --reload
```

Send a request to 'localhost:8000/cdr_daily_spread' to check if it works.

## Edit code using agents

Let's assume we want our agents to extend the functionality of the application. Let's modify the endpoint to check the spread also for 11bits, another gamedev company, and compare it for both stocks. Also, let's separate the internal logic into a different file.

Finally, instantiate a chat between the Engineer and the Admin. It will start by exploring the filesystem first, and after that, it will wait for our orders. Then, we will explain the task to the Engineer and ask him to provide a plan of changes first - according to my experience, that greatly increases the quality of LLM responses.

After that, introduce changes with the Engineer one after another. Ask him to correct something or improve the functionality if needed. Do not hesitate to interrupt the tool's execution if you feel he is going to do something wrong. If errors occur, provide him with the error log and ask him to check out the file to refresh his knowledge about it before actually introducing changes.
"""
logger.info("## Prepare code to work with")

chat_result = user_proxy.initiate_chat(
    manager,
    message="""
You will need to improve app in FastApi. For now, check out all the application files, try to understand it and wait for next instructions.
""",
)

"""
## Result

Finally, our agents modified a code so it looks like this:

```python
# backend_dir/main.py


app = FastAPI()

@app.get("/compare_daily_spread")
async def compare_daily_spread():
    cdr_spread = calculate_daily_spread("CDR.WA")
    bits_spread = calculate_daily_spread("11B.WA")
    spread_difference = cdr_spread - bits_spread
    if spread_difference > 0:
        return {'message': 'CD Project Red has a larger daily spread', 'difference': spread_difference}
    elif spread_difference < 0:
        return {'message': '11bits Studio has a larger daily spread', 'difference': -spread_difference}
    else:
        return {'message': 'Both stocks have the same daily spread', 'difference': 0}


# backend_dir/spread_calculation.py


def calculate_daily_spread(ticker):
    stock = yf.Ticker(ticker)
    today_data = stock.history(period="1d")
    spread = ((today_data["High"] - today_data["Low"]) / today_data["Low"]) * 100
    return spread.values[0]
```

You can check out work of application with Postman or curl and see the next output:

```json
{
    "message": "11bits Studio has a larger daily spread",
    "difference": 1.7968083865943187
}
```
"""
logger.info("## Result")

logger.info("\n\n[DONE]", bright=True)