from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.tools.reddit_search.tool import RedditSearchSchema
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
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
# Reddit Search 

In this notebook, we learn how the Reddit search tool works.  
First make sure that you have installed praw with the command below:
"""
logger.info("# Reddit Search")

# %pip install --upgrade --quiet  praw

"""
Then you need to set you need to set up the proper API keys and environment variables. You would need to create a Reddit user account and get credentials. So, create a Reddit user account by going to https://www.reddit.com  and signing up.  
Then get your credentials by going to https://www.reddit.com/prefs/apps and creating an app.  
You should have your client_id and secret from creating the app. Now, you can paste those strings in client_id and client_secret variable.  
Note: You can put any string for user_agent
"""
logger.info("Then you need to set you need to set up the proper API keys and environment variables. You would need to create a Reddit user account and get credentials. So, create a Reddit user account by going to https://www.reddit.com  and signing up.")

client_id = ""
client_secret = ""
user_agent = ""


search = RedditSearchRun(
    api_wrapper=RedditSearchAPIWrapper(
        reddit_client_id=client_id,
        reddit_client_secret=client_secret,
        reddit_user_agent=user_agent,
    )
)

"""
You can then set your queries for example, what subreddit you want to query, how many posts you want to be returned, how you would like the result to be sorted etc.
"""
logger.info("You can then set your queries for example, what subreddit you want to query, how many posts you want to be returned, how you would like the result to be sorted etc.")


search_params = RedditSearchSchema(
    query="beginner", sort="new", time_filter="week", subreddit="python", limit="2"
)

"""
Finally run the search and get your results
"""
logger.info("Finally run the search and get your results")

result = search.run(tool_input=search_params.dict())

logger.debug(result)

"""
Here is an example of printing the result.  
Note: You may get different output depending on the newest post in the subreddit but the formatting should be similar.

> Searching r/python found 2 posts:
> Post Title: 'Setup Github Copilot in Visual Studio Code'
> User: Feisty-Recording-715
> Subreddit: r/Python:
>                     Text body: 🛠️ This tutorial is perfect for beginners looking to strengthen their understanding of version control or for experienced developers seeking a quick reference for GitHub setup in Visual Studio Code.
>
>🎓 By the end of this video, you'll be equipped with the skills to confidently manage your codebase, collaborate with others, and contribute to open-source projects on GitHub.
>
>
>Video link: https://youtu.be/IdT1BhrSfdo?si=mV7xVpiyuhlD8Zrw
>
>Your feedback is welcome
>                     Post URL: https://www.reddit.com/r/Python/comments/1823wr7/setup_github_copilot_in_visual_studio_code/
>                     Post Category: N/A.
>                     Score: 0
>
>Post Title: 'A Chinese Checkers game made with pygame and PySide6, with custom bots support'
>User: HenryChess
>Subreddit: r/Python:
>                     Text body: GitHub link: https://github.com/henrychess/pygame-chinese-checkers
>
>I'm not sure if this counts as beginner or intermediate. I think I'm still in the beginner zone, so I flair it as beginner.
>
>This is a Chinese Checkers (aka Sternhalma) game for 2 to 3 players. The bots I wrote are easy to beat, as they're mainly for debugging the game logic part of the code. However, you can write up your own custom bots. There is a guide at the github page.
>                     Post URL: https://www.reddit.com/r/Python/comments/181xq0u/a_chinese_checkers_game_made_with_pygame_and/
>                     Post Category: N/A.
 >                    Score: 1

## Using tool with an agent chain

Reddit search functionality is also provided as a multi-input tool. In this example, we adapt [existing code from the docs](https://python.langchain.com/v0.1/docs/modules/memory/agent_with_memory/), and use ChatOllama to create an agent chain with memory. This agent chain is able to pull information from Reddit and use these posts to respond to subsequent input. 

To run the example, add your reddit API access information and also get an Ollama key from the [Ollama API](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-api-key).
"""
logger.info("## Using tool with an agent chain")


client_id = ""
client_secret = ""
user_agent = ""
ollama_

template = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

tools = [
    RedditSearchRun(
        api_wrapper=RedditSearchAPIWrapper(
            reddit_client_id=client_id,
            reddit_client_secret=client_secret,
            reddit_user_agent=user_agent,
        )
    )
]

prompt = StructuredChatAgent.create_prompt(
    prefix=prefix,
    tools=tools,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

llm = ChatOllama(model="llama3.2")

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = StructuredChatAgent(llm_chain=llm_chain, verbose=True, tools=tools)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, verbose=True, memory=memory, tools=tools
)

agent_chain.run(input="What is the newest post on r/langchain for the week?")
agent_chain.run(input="Who is the author of the post?")

logger.info("\n\n[DONE]", bright=True)