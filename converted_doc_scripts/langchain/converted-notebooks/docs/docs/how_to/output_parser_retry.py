from jet.adapters.langchain.chat_ollama import ChatOllama, Ollama
from jet.logger import logger
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field
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
# How to retry when a parsing error occurs

While in some cases it is possible to fix any parsing mistakes by only looking at the output, in other cases it isn't. An example of this is when the output is not just in the incorrect format, but is partially complete. Consider the below example.
"""
logger.info("# How to retry when a parsing error occurs")


template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""


class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


parser = PydanticOutputParser(pydantic_object=Action)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt_value = prompt.format_prompt(query="who is leo di caprios gf?")

bad_response = '{"action": "search"}'

"""
If we try to parse this response as is, we will get an error:
"""
logger.info("If we try to parse this response as is, we will get an error:")

try:
    parser.parse(bad_response)
except OutputParserException as e:
    logger.debug(e)

"""
If we try to use the `OutputFixingParser` to fix this error, it will be confused - namely, it doesn't know what to actually put for action input.
"""
logger.info("If we try to use the `OutputFixingParser` to fix this error, it will be confused - namely, it doesn't know what to actually put for action input.")

fix_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOllama(model="llama3.2"))

fix_parser.parse(bad_response)

"""
Instead, we can use the RetryOutputParser, which passes in the prompt (as well as the original output) to try again to get a better response.
"""
logger.info("Instead, we can use the RetryOutputParser, which passes in the prompt (as well as the original output) to try again to get a better response.")


retry_parser = RetryOutputParser.from_llm(parser=parser, llm=Ollama(temperature=0))

retry_parser.parse_with_prompt(bad_response, prompt_value)

"""
We can also add the RetryOutputParser easily with a custom chain which transform the raw LLM/ChatModel output into a more workable format.
"""
logger.info("We can also add the RetryOutputParser easily with a custom chain which transform the raw LLM/ChatModel output into a more workable format.")


completion_chain = prompt | Ollama(temperature=0)

main_chain = RunnableParallel(
    completion=completion_chain, prompt_value=prompt
) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))


main_chain.invoke({"query": "who is leo di caprios gf?"})

"""
Find out api documentation for [RetryOutputParser](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.retry.RetryOutputParser.html#langchain.output_parsers.retry.RetryOutputParser).
"""
logger.info("Find out api documentation for [RetryOutputParser](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.retry.RetryOutputParser.html#langchain.output_parsers.retry.RetryOutputParser).")


logger.info("\n\n[DONE]", bright=True)