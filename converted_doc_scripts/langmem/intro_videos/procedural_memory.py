import os
from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_store
from langmem import create_prompt_optimizer
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_store
from langgraph_supervisor import create_supervisor
from langmem import create_multi_prompt_optimizer
from langmem import Prompt

    
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

initialize_ollama_settings()

logger.orange("""## Procedural memory""")

# %%capture cap --no-stderr
# %pip install -U langmem langgraph


store = InMemoryStore()
store.put(("instructions",), key="agent_instructions", value={"prompt": "Write good emails."})


def draft_email(to: str, subject: str, body: str):
    """Submit an email draft."""
    return "Draft saved succesfully."

def prompt(state):
    item = store.get(("instructions",), key="agent_instructions")
    instructions = item.value["prompt"]
    sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
    return [sys_prompt] + state['messages']

agent = create_react_agent("ollama:llama3.1", prompt=prompt, tools=[draft_email], store=store)

result = agent.invoke(
    {"messages": [
        {"role": "user", "content" :"Draft an email to joe@langchain.dev saying that we want to schedule a followup meeting for thursday at noon."}]}
)
result['messages'][1].pretty_logger.debug()

logger.orange("""# Update the prompt""")


optimizer = create_prompt_optimizer("ollama:llama3.1")

current_prompt = store.get(("instructions",), key="agent_instructions").value["prompt"]
feedback = {"request": "Always sign off from 'William'; for meeting requests, offer to schedule on Zoom or Google Meet"}

optimizer_result = optimizer.invoke({"prompt": current_prompt, "trajectories": [(result["messages"], feedback)]})

logger.debug(optimizer_result)

store.put(("instructions",), key="agent_instructions", value={"prompt": optimizer_result})

result = agent.invoke(
    {"messages": [
        {"role": "user", "content" :"Draft an email to joe@langchain.dev saying that we want to schedule a followup meeting for thursday at noon."}]}
)
result['messages'][1].pretty_logger.debug()

result = agent.invoke(
    {"messages": [
        {"role": "user", "content" : "Let roger@langchain.dev know that the release should be later by 4:00 PM."}]}
)
result['messages'][1].pretty_logger.debug()

logger.orange("""## Multi-agent""")

# %%capture cap --no-stderr
# %pip install -U langgraph-supervisor


store = InMemoryStore()

store.put(("instructions",), key="email_agent", value={"prompt": "Write good emails. Repeat your draft content to the user after submitting."})
store.put(("instructions",), key="twitter_agent", value={"prompt": "Write fire tweets. Repeat the tweet content to the user upon submission."})

def draft_email(to: str, subject: str, body: str):
    """Submit an email draft."""
    return "Draft saved succesfully."

def prompt_email(state):
    item = store.get(("instructions",), key="email_agent")
    instructions = item.value["prompt"]
    sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
    return [sys_prompt] + state['messages']

email_agent = create_react_agent(
    "ollama:llama3.1", 
    prompt=prompt_email, 
    tools=[draft_email], 
    store=store,
    name="email_assistant",
)


def tweet(to: str, subject: str, body: str):
    """Poast a tweet."""
    return "Legendary."

def prompt_social_media(state):
    item = store.get(("instructions",), key="twitter_agent")
    instructions = item.value["prompt"]
    sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
    return [sys_prompt] + state['messages']

social_media_agent = create_react_agent(
    "ollama:llama3.1", 
    prompt=prompt_social_media, 
    tools=[tweet], 
    store=store,
    name="social_media_agent",
)


workflow = create_supervisor(
    [email_agent, social_media_agent],
    model="ollama:llama3.1",
    prompt=(
        "You are a team supervisor managing email and tweet assistants to help with correspondance."
    )
)

app = workflow.compile(store=store)

result = app.invoke(
    {"messages": [
        {"role": "user", "content" :"Draft an email to joe@langchain.dev saying that we want to schedule a followup meeting for thursday at noon."}]},
)

result["messages"][3].pretty_logger.debug()


feedback = {"request": "Always sign off emails from 'William'; for meeting requests, offer to schedule on Zoom or Google Meet"}

optimizer = create_multi_prompt_optimizer("ollama:llama3.1")

?Prompt

email_prompt = store.get(("instructions",), key="email_agent").value['prompt']
tweet_prompt = store.get(("instructions",), key="twitter_agent").value['prompt']

email_prompt = {
    "name": "email_prompt",
    "prompt": email_prompt,
    "when_to_update": "Only if feedback is provided indicating email writing performance needs improved."
}
tweet_prompt = {
    "name": "tweet_prompt",
    "prompt": tweet_prompt,
    "when_to_update": "Only if tweet writing generation needs improvement."
}


optimizer_result = optimizer.invoke({"prompts": [tweet_prompt, email_prompt], "trajectories": [(result["messages"], feedback)]})

optimizer_result

store.put(("instructions",), key="email_agent", value={"prompt": optimizer_result[1]['prompt']})

result = app.invoke(
    {"messages": [
        {"role": "user", "content" :"Draft an email to joe@langchain.dev saying that we want to schedule a followup meeting for thursday at noon."}]},
)

result["messages"][3].pretty_logger.debug()

logger.info("\n\n[DONE]", bright=True)