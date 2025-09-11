from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.llms import AmazonAPIGateway
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
# Amazon API Gateway

>[Amazon API Gateway](https://aws.amazon.com/api-gateway/) is a fully managed service that makes it easy for developers to create, publish, maintain, monitor, and secure APIs at any >scale. APIs act as the "front door" for applications to access data, business logic, or functionality from your backend services. Using `API Gateway`, you can create RESTful APIs and >WebSocket APIs that enable real-time two-way communication applications. API Gateway supports containerized and serverless workloads, as well as web applications.

>`API Gateway` handles all the tasks involved in accepting and processing up to hundreds of thousands of concurrent API calls, including traffic management, CORS support, authorization >and access control, throttling, monitoring, and API version management. `API Gateway` has no minimum fees or startup costs. You pay for the API calls you receive and the amount of data >transferred out and, with the `API Gateway` tiered pricing model, you can reduce your cost as your API usage scales.
"""
logger.info("# Amazon API Gateway")

# %pip install -qU langchain-community

"""
## LLM
"""
logger.info("## LLM")


api_url = "https://<api_gateway_id>.execute-api.<region>.amazonaws.com/LATEST/HF"
llm = AmazonAPIGateway(api_url=api_url)

parameters = {
    "max_new_tokens": 100,
    "num_return_sequences": 1,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": False,
    "return_full_text": True,
    "temperature": 0.2,
}

prompt = "what day comes after Friday?"
llm.model_kwargs = parameters
llm(prompt)

"""
## Agent
"""
logger.info("## Agent")


parameters = {
    "max_new_tokens": 50,
    "num_return_sequences": 1,
    "top_k": 250,
    "top_p": 0.25,
    "do_sample": False,
    "temperature": 0.1,
}

llm.model_kwargs = parameters

tools = load_tools(["python_repl", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run(
    """
Write a Python script that prints "Hello, world!"
"""
)

result = agent.run(
    """
What is 2.3 ^ 4.5?
"""
)

result.split("\n")[0]

logger.info("\n\n[DONE]", bright=True)