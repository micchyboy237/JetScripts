from jet.llm.ollama.base import Ollama
from jet.logger import logger
from llama_index.core.prompts.base import PromptTemplate


template = PromptTemplate("""You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {query} \n
Output (4 queries):""")

llm = Ollama(model="llama3.2")

query = "List trending isekai anime today."

message = template.format(query=query)
response1 = llm.chat(message)
logger.debug("Response:", response1.message.content)
