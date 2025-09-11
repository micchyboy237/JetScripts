from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_ollama_functions_agent
from langchain_community.tools.passio_nutrition_ai import NutritionAI
from langchain_community.utilities.passio_nutrition_ai import NutritionAIAPI
from langchain_core.utils import get_from_env
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
# Passio NutritionAI

To best understand how NutritionAI can give your agents super food-nutrition powers, let's build an agent that can find that information via Passio NutritionAI.

## Define tools

We first need to create [the Passio NutritionAI tool](/docs/integrations/tools/passio_nutrition_ai).

### [Passio Nutrition AI](/docs/integrations/tools/passio_nutrition_ai)

We have a built-in tool in LangChain to easily use Passio NutritionAI to find food nutrition facts.
Note that this requires an API key - they have a free tier.

Once you create your API key, you will need to export that as:

```bash
export NUTRITIONAI_SUBSCRIPTION_KEY="..."
```

... or provide it to your Python environment via some other means such as the `dotenv` package.  You an also explicitly control the key via constructor calls.
"""
logger.info("# Passio NutritionAI")


load_dotenv()

nutritionai_subscription_key = get_from_env(
    "nutritionai_subscription_key", "NUTRITIONAI_SUBSCRIPTION_KEY"
)


nutritionai_search = NutritionAI(api_wrapper=NutritionAIAPI())

nutritionai_search.invoke("chicken tikka masala")

nutritionai_search.invoke("Schnuck Markets sliced pepper jack cheese")

"""
### Tools

Now that we have the tool, we can create a list of tools that we will use downstream.
"""
logger.info("### Tools")

tools = [nutritionai_search]

"""
## Create the agent

Now that we have defined the tools, we can create the agent. We will be using an Ollama Functions agent - for more information on this type of agent, as well as other options, see [this guide](/docs/concepts/agents)

First, we choose the LLM we want to be guiding the agent.
"""
logger.info("## Create the agent")


llm = ChatOllama(model="llama3.2")

"""
Next, we choose the prompt we want to use to guide the agent.
"""
logger.info("Next, we choose the prompt we want to use to guide the agent.")


prompt = hub.pull("hwchase17/ollama-functions-agent")
prompt.messages

"""
Now, we can initialize the agent with the LLM, the prompt, and the tools. The agent is responsible for taking in input and deciding what actions to take. Crucially, the Agent does not execute those actions - that is done by the AgentExecutor (next step). For more information about how to think about these components, see our [conceptual guide](/docs/concepts/agents)
"""
logger.info("Now, we can initialize the agent with the LLM, the prompt, and the tools. The agent is responsible for taking in input and deciding what actions to take. Crucially, the Agent does not execute those actions - that is done by the AgentExecutor (next step). For more information about how to think about these components, see our [conceptual guide](/docs/concepts/agents)")


agent = create_ollama_functions_agent(llm, tools, prompt)

"""
Finally, we combine the agent (the brains) with the tools inside the AgentExecutor (which will repeatedly call the agent and execute tools). For more information about how to think about these components, see our [conceptual guide](/docs/concepts/agents)
"""
logger.info("Finally, we combine the agent (the brains) with the tools inside the AgentExecutor (which will repeatedly call the agent and execute tools). For more information about how to think about these components, see our [conceptual guide](/docs/concepts/agents)")


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

"""
## Run the agent

We can now run the agent on a few queries! Note that for now, these are all **stateless** queries (it won't remember previous interactions).
"""
logger.info("## Run the agent")

agent_executor.invoke({"input": "hi!"})

agent_executor.invoke({"input": "how many calories are in a slice pepperoni pizza?"})

"""
If we want to keep track of these messages automatically, we can wrap this in a RunnableWithMessageHistory. For more information on how to use this, see [this guide](/docs/how_to/message_history)
"""
logger.info("If we want to keep track of these messages automatically, we can wrap this in a RunnableWithMessageHistory. For more information on how to use this, see [this guide](/docs/how_to/message_history)")

agent_executor.invoke(
    {"input": "I had bacon and eggs for breakfast.  How many calories is that?"}
)

agent_executor.invoke(
    {
        "input": "I had sliced pepper jack cheese for a snack.  How much protein did I have?"
    }
)

agent_executor.invoke(
    {
        "input": "I had sliced colby cheese for a snack. Give me calories for this Schnuck Markets product."
    }
)

agent_executor.invoke(
    {
        "input": "I had chicken tikka masala for dinner.  how much calories, protein, and fat did I have with default quantity?"
    }
)

"""
## Conclusion

That's a wrap! In this quick start we covered how to create a simple agent that is able to incorporate food-nutrition information into its answers. Agents are a complex topic, and there's lot to learn!


"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)