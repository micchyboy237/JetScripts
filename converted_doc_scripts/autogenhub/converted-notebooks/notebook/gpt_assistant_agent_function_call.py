from autogen import UserProxyAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.function_utils import get_function_schema
from jet.logger import CustomLogger
from typing import Annotated, Literal
import autogen
import os
import requests

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# From Dad Jokes To Sad Jokes: Function Calling with GPTAssistantAgent

Autogen allows `GPTAssistantAgent` to be augmented with "tools" — pre-defined functions or capabilities — that extend its ability to handle specific tasks, similar to how one might natively utilize tools in the [Ollama Assistant's API](https://platform.openai.com/docs/assistants/tools).

In this notebook, we create a basic Multi-Agent System using Autogen's `GPTAssistantAgent` to convert Dad jokes on a specific topic into Sad jokes. It consists of a "Dad" agent which has the ability to search the [Dad Joke API](https://icanhazdadjoke.com/api) and a "Sad Joker" agent which converts the Dad jokes into Sad jokes. The Sad Joker then writes the sad jokes into a txt file.

In this process we demonstrate how to call tools and perform function calling for `GPTAssistantAgent`.

## Requirements
AutoGen requires Python 3.8 or newer. For this notebook, please install `autogen`:
"""
logger.info("# From Dad Jokes To Sad Jokes: Function Calling with GPTAssistantAgent")

# %pip install autogen

"""
Import Dependencies
"""
logger.info("Import Dependencies")




config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
)

"""
## Creating the Functions
We need to create functions for our Agents to call.

This function calls the Dad Joke API with a search term that the agent creates and returns a list of dad jokes.
"""
logger.info("## Creating the Functions")

def get_dad_jokes(search_term: str, page: int = 1, limit: int = 10) -> str:
    """
    Fetches a list of dad jokes based on a search term.

    Parameters:
    - search_term: The search term to find jokes about.
    - page: The page number of results to fetch (default is 1).
    - limit: The number of results to return per page (default is 20, max is 30).

    Returns:
    A list of dad jokes.
    """
    url = "https://icanhazdadjoke.com/search"
    headers = {"Accept": "application/json"}
    params = {"term": search_term, "page": page, "limit": limit}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        jokes = [joke["joke"] for joke in data["results"]]
        return jokes
    else:
        return f"Failed to fetch jokes, status code: {response.status_code}"

jokes = get_dad_jokes("cats")
logger.debug(jokes)

"""
This function allows the Agents to write to a txt file.
"""
logger.info("This function allows the Agents to write to a txt file.")

def write_to_txt(content: str, filename: str = "dad_jokes.txt"):
    """
    Writes a formatted string to a text file.
    Parameters:

    - content: The formatted string to write.
    - filename: The name of the file to write to. Defaults to "output.txt".
    """
    with open(filename, "w") as file:
        file.write(content)

content = "\n".join(jokes)  # Format the jokes from the above example
write_to_txt(content)

"""
## Create Function Schemas
In order to use the functions within our GPTAssistantAgents, we need to generate function schemas. This can be done by using `get_function_schema`
"""
logger.info("## Create Function Schemas")

get_dad_jokes_schema = get_function_schema(
    get_dad_jokes,
    name="get_dad_jokes",
    description="Fetches a list of dad jokes based on a search term. Allows pagination with page and limit parameters.",
)

write_to_txt_schema = get_function_schema(
    write_to_txt,
    name="write_to_txt",
    description="Writes a formatted string to a text file. If the file does not exist, it will be created. If the file does exist, it will be overwritten.",
)

"""
## Creating the Agents
In this section we create and configure our Dad and Sad Joker Agents

### Set up the User Proxy
"""
logger.info("## Creating the Agents")

user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)

"""
### The Dad Agent
We create the Dad agent using `GPTAssistantAgent`, in order for us to enable the Dad to use the `get_dad_jokes` function we need to provide it the function's specification in our `llm_config`.

We format the `tools` within our `llm_config` in the same format as provided in the [Ollama Assistant tools docs](https://platform.openai.com/docs/assistants/tools/function-calling).
"""
logger.info("### The Dad Agent")

the_dad = GPTAssistantAgent(
    name="the_dad",
    instructions="""
    As 'The Dad', your primary role is to entertain by fetching dad jokes which the sad joker will transform into 'sad jokes' based on a given theme. When provided with a theme, such as 'plants' or 'animals', your task is as follows:

    1. Use the 'get_dad_jokes' function to search for dad jokes related to the provided theme by providing a search term related to the theme. Fetch a list of jokes that are relevant to the theme.
    2. Present these jokes to the sad joker in a format that is clear and easy to read, preparing them for transformation.

    Remember, the team's goal is to creatively adapt the essence of each dad joke to fit the 'sad joke' format, all while staying true to the theme provided by the user.
    """,
    overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
    overwrite_tools=True,  # overwrite any existing tools with the ones provided
    llm_config={
        "config_list": config_list,
        "tools": [get_dad_jokes_schema],
    },
)

"""
Next, we register the `get_dad_jokes` function with the Dad `GPTAssistantAgent`
"""
logger.info("Next, we register the `get_dad_jokes` function with the Dad `GPTAssistantAgent`")

the_dad.register_function(
    function_map={
        "get_dad_jokes": get_dad_jokes,
    },
)

"""
### The Sad Joker Agent
We then create and configure the Sad Joker agent in a similar manner to the Dad agent above.
"""
logger.info("### The Sad Joker Agent")

the_sad_joker = GPTAssistantAgent(
    name="the_sad_joker",
    instructions="""
    As 'The Sad Joker', your unique role is to take dad jokes and creatively transform them into 'sad jokes'. When you receive a list of dad jokes, themed around topics like 'plants' or 'animals', you should:

    1. Read through each dad joke carefully, understanding its theme and punchline.
    2. Creatively alter the joke to change its mood from humorous to somber or melancholic. This may involve tweaking the punchline, modifying the setup, or even completely reimagining the joke while keeping it relevant to the original theme.
    3. Ensure your transformations maintain a clear connection to the original theme and are understandable as adaptations of the dad jokes provided.
    4. Write your transformed sad jokes to a text file using the 'write_to_txt' function. Use meaningful file names that reflect the theme or the nature of the jokes within, unless a specific filename is requested.

    Your goal is not just to alter the mood of the jokes but to do so in a way that is creative, thoughtful, and respects the essence of the original humor. Remember, while the themes might be light-hearted, your transformations should offer a melancholic twist that makes them uniquely 'sad jokes'.
    """,
    overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
    overwrite_tools=True,  # overwrite any existing tools with the ones provided
    llm_config={
        "config_list": config_list,
        "tools": [write_to_txt_schema],
    },
)

"""
Register the `write_to_txt` function with the Sad Joker `GPTAssistantAgent`
"""
logger.info("Register the `write_to_txt` function with the Sad Joker `GPTAssistantAgent`")

the_sad_joker.register_function(
    function_map={
        "write_to_txt": write_to_txt,
    },
)

"""
## Creating the Groupchat and Starting the Conversation

Create the groupchat
"""
logger.info("## Creating the Groupchat and Starting the Conversation")

groupchat = autogen.GroupChat(agents=[user_proxy, the_dad, the_sad_joker], messages=[], max_round=15)
group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

"""
Start the Conversation
"""
logger.info("Start the Conversation")

user_proxy.initiate_chat(group_chat_manager, message="Jokes about cats")

logger.info("\n\n[DONE]", bright=True)