from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_ollama_tools_agent
from langchain_core.tools import tool
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
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
# Log, Trace, and Monitor

When building apps or agents using Langchain, you end up making multiple API calls to fulfill a single user request. However, these requests are not chained when you want to analyse them. With [**Portkey**](/docs/integrations/providers/portkey/), all the embeddings, completions, and other requests from a single user request will get logged and traced to a common ID, enabling you to gain full visibility of user interactions.

This notebook serves as a step-by-step guide on how to log, trace, and monitor Langchain LLM calls using `Portkey` in your Langchain app.

First, let's import Portkey, Ollama, and Agent tools
"""
logger.info("# Log, Trace, and Monitor")



"""
Paste your Ollama API key below. [(You can find it here)](https://platform.ollama.com/account/api-keys)
"""
logger.info("Paste your Ollama API key below. [(You can find it here)](https://platform.ollama.com/account/api-keys)")

# os.environ["OPENAI_API_KEY"] = "..."

"""
## Get Portkey API Key
1. Sign up for [Portkey here](https://app.portkey.ai/signup)
2. On your [dashboard](https://app.portkey.ai/), click on the profile icon on the bottom left, then click on "Copy API Key"
3. Paste it below
"""
logger.info("## Get Portkey API Key")

PORTKEY_API_KEY = "..."  # Paste your Portkey API Key here

"""
## Set Trace ID
1. Set the trace id for your request below
2. The Trace ID can be common for all API calls originating from a single request
"""
logger.info("## Set Trace ID")

TRACE_ID = "uuid-trace-id"  # Set trace id here

"""
## Generate Portkey Headers
"""
logger.info("## Generate Portkey Headers")

portkey_headers = createHeaders(
    api_key=PORTKEY_API_KEY, provider="ollama", trace_id=TRACE_ID
)

"""
Define the prompts and the tools to use
"""
logger.info("Define the prompts and the tools to use")


prompt = hub.pull("hwchase17/ollama-tools-agent")


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


tools = [multiply, exponentiate]

"""
Run your agent as usual. The **only** change is that we will **include the above headers** in the request now.
"""
logger.info("Run your agent as usual. The **only** change is that we will **include the above headers** in the request now.")

model = ChatOllama(
    base_url=PORTKEY_GATEWAY_URL, default_headers=portkey_headers, temperature=0
)

agent = create_ollama_tools_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "Take 3 to the fifth power and multiply that by thirty six, then square the result"
    }
)

"""
## How Logging & Tracing Works on Portkey

**Logging**
- Sending your request through Portkey ensures that all of the requests are logged by default
- Each request log contains `timestamp`, `model name`, `total cost`, `request time`, `request json`, `response json`, and additional Portkey features

**[Tracing](https://portkey.ai/docs/product/observability-modern-monitoring-for-llms/traces)**
- Trace id is passed along with each request and is visible on the logs on Portkey dashboard
- You can also set a **distinct trace id** for each request if you want
- You can append user feedback to a trace id as well. [More info on this here](https://portkey.ai/docs/product/observability-modern-monitoring-for-llms/feedback)

For the above request, you will be able to view the entire log trace like this
![View Langchain traces on Portkey](https://assets.portkey.ai/docs/agent_tracing.gif)

## Advanced LLMOps Features - Caching, Tagging, Retries

In addition to logging and tracing, Portkey provides more features that add production capabilities to your existing workflows:

**Caching**

Respond to previously served customers queries from cache instead of sending them again to Ollama. Match exact strings OR semantically similar strings. Cache can save costs and reduce latencies by 20x. [Docs](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic)

**Retries**

Automatically reprocess any unsuccessful API requests **`upto 5`** times. Uses an **`exponential backoff`** strategy, which spaces out retry attempts to prevent network overload. [Docs](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations)

**Tagging**

Track and audit each user interaction in high detail with predefined tags. [Docs](https://portkey.ai/docs/product/observability-modern-monitoring-for-llms/metadata)
"""
logger.info("## How Logging & Tracing Works on Portkey")

logger.info("\n\n[DONE]", bright=True)