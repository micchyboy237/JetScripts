from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_aws import ChatBedrockConverse
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
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
# Response metadata

Many model providers include some metadata in their chat generation [responses](/docs/concepts/messages/#aimessage). This metadata can be accessed via the `AIMessage.response_metadata: Dict` attribute. Depending on the model provider and model configuration, this can contain information like [token counts](/docs/how_to/chat_token_usage_tracking), [logprobs](/docs/how_to/logprobs), and more.

Here's what the response metadata looks like for a few different providers:

## Ollama
"""
logger.info("# Response metadata")


llm = ChatOllama(model="llama3.2")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata

"""
## Ollama
"""
logger.info("## Ollama")


llm = ChatOllama(model="llama3.2")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata

"""
## Google Generative AI
"""
logger.info("## Google Generative AI")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata

"""
## Bedrock (Ollama)
"""
logger.info("## Bedrock (Ollama)")


llm = ChatBedrockConverse(model="anthropic.claude-3-7-sonnet-20250219-v1:0")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata

"""
## MistralAI
"""
logger.info("## MistralAI")


llm = ChatMistralAI(model="mistral-small-latest")
msg = llm.invoke([("human", "What's the oldest known example of cuneiform")])
msg.response_metadata

"""
## Groq
"""
logger.info("## Groq")


llm = ChatGroq(model="llama-3.1-8b-instant")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata

"""
## FireworksAI
"""
logger.info("## FireworksAI")


llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")
msg = llm.invoke("What's the oldest known example of cuneiform")
msg.response_metadata

logger.info("\n\n[DONE]", bright=True)