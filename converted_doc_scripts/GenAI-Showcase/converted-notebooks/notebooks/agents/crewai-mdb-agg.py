from crewai import Agent, Crew, Process, Task
from jet.llm.ollama.base_langchain import AzureChatOllama
from jet.logger import CustomLogger
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
import os
import pprint
import pymongo
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

pip install pymongo==4.7.2 crewai==0.22.5 langchain==0.1.10 langchain-community langchain-ollama==0.0.5 duckduckgo-search==6.1.5



MDB_URI = "mongodb+srv://<user>:<password>@cluster0.abc123.mongodb.net/"
client = pymongo.MongoClient(MDB_URI, appname="devrel.showcase.crewai")
db = client["sample_analytics"]
collection = db["transactions"]


AZURE_OPENAI_ENDPOINT = "https://__DEMO__.ollama.azure.com"
# AZURE_OPENAI_API_KEY = "__AZURE_OPENAI_API_KEY__"
deployment_name = "gpt-4-32k"  # The name of your model deployment
default_llm = AzureChatOllama(
    ollama_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2023-07-01-preview"),
    azure_deployment=deployment_name,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY,
)


duck_duck_go = DuckDuckGoSearchResults(backend="news", max_results=10)

@tool
def search_tool(query: str):
    """
    Perform online research on a particular stock.
    Will return search results along with snippets of each result.
    """
    logger.debug("\n\nSearching DuckDuckGo for:", query)
    search_results = duck_duck_go.run(query)
    search_results_str = "[recent news for: " + query + "]\n" + str(search_results)
    return search_results_str


AGENT_ROLE = "Investment Researcher"
AGENT_GOAL = """
  Research stock market trends, company news, and analyst reports to identify potential investment opportunities.
"""
researcher = Agent(
    role=AGENT_ROLE,
    goal=AGENT_GOAL,
    verbose=True,
    llm=default_llm,
    backstory="Expert stock researcher with decades of experience.",
    tools=[search_tool],
)

task1 = Task(
    description="""
Using the following information:

[VERIFIED DATA]
{agg_data}

*note*
The data represents the net gain or loss of each stock symbol for each transaction type (buy/sell).
Net gain or loss is a crucial metric used to gauge the profitability or efficiency of an investment.
It's computed by subtracting the total buy value from the total sell value for each stock.
[END VERIFIED DATA]

[TASK]
- Generate a detailed financial report of the VERIFIED DATA.
- Research current events and trends, and provide actionable insights and recommendations.


[report criteria]
  - Use all available information to prepare this final financial report
  - Include a TLDR summary
  - Include 'Actionable Insights'
  - Include 'Strategic Recommendations'
  - Include a 'Other Observations' section
  - Include a 'Conclusion' section
  - IMPORTANT! You are a friendly and helpful financial expert. Always provide the best possible answer using the available information.
[end report criteria]
  """,
    agent=researcher,
    expected_output="concise markdown financial summary of the verified data and list of key points and insights from researching current events",
    tools=[search_tool],
)
tech_crew = Crew(agents=[researcher], tasks=[task1], process=Process.sequential)

pipeline = [
    {
        "$unwind": "$transactions"  # Deconstruct the transactions array into separate documents
    },
    {
        "$group": {  # Group documents by stock symbol
            "_id": "$transactions.symbol",  # Use symbol as the grouping key
            "buyValue": {  # Calculate total buy value
                "$sum": {
                    "$cond": [  # Conditional sum based on transaction type
                        {
                            "$eq": ["$transactions.transaction_code", "buy"]
                        },  # Check for "buy" transactions
                        {
                            "$toDouble": "$transactions.total"
                        },  # Convert total to double for sum
                        0,  # Default value for non-buy transactions
                    ]
                }
            },
            "sellValue": {  # Calculate total sell value (similar to buyValue)
                "$sum": {
                    "$cond": [
                        {"$eq": ["$transactions.transaction_code", "sell"]},
                        {"$toDouble": "$transactions.total"},
                        0,
                    ]
                }
            },
        }
    },
    {
        "$project": {  # Project desired fields (renaming and calculating net gain)
            "_id": 0,  # Exclude original _id field
            "symbol": "$_id",  # Rename _id to symbol for clarity
            "netGain": {"$subtract": ["$sellValue", "$buyValue"]},  # Calculate net gain
        }
    },
    {
        "$sort": {"netGain": -1}  # Sort results by net gain (descending)
    },
    {"$limit": 3},  # Limit results to top 3 stocks
]
results = list(collection.aggregate(pipeline))
client.close()

logger.debug("MongoDB Aggregation Pipeline Results:")

pprint.plogger.debug(
    results
)  # pprint is used to  to “pretty-print” arbitrary Python data structures

tech_crew.kickoff(inputs={"agg_data": str(results)})

logger.info("\n\n[DONE]", bright=True)