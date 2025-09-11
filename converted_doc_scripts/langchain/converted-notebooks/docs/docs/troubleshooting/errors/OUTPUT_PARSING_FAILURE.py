from jet.logger import logger
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
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
# OUTPUT_PARSING_FAILURE

An [output parser](/docs/concepts/output_parsers) was unable to handle model output as expected.

To illustrate this, let's say you have an output parser that expects a chat model to output JSON surrounded by a markdown code tag (triple backticks). Here would be an example of good input:
"""
logger.info("# OUTPUT_PARSING_FAILURE")


message = AIMessage(content='```\n{"foo": "bar"}\n```')
output_parser = JsonOutputParser()
output_parser.invoke(message)

"""
Internally, our JSON parser stripped out the markdown fence and newlines and then ran `json.loads`.

If instead the chat model generated an output with malformed JSON, we will get an error:
"""
logger.info("Internally, our JSON parser stripped out the markdown fence and newlines and then ran `json.loads`.")

message = AIMessage(content='```\n{{"foo":\n```')
output_parser = JsonOutputParser()
output_parser.invoke(message)

"""
Note that some prebuilt constructs like [legacy LangChain agents](/docs/how_to/agent_executor) and chains may use output parsers internally,
so you may see this error even if you're not visibly instantiating and using an output parser.

## Troubleshooting

The following may help resolve this error:

- Consider using [tool calling or other structured output techniques](/docs/how_to/structured_output/) if possible without an output parser to reliably output parseable values.
  - If you are using a prebuilt chain or agent, use [LangGraph](https://langchain-ai.github.io/langgraph/) to compose your logic explicitly instead.
- Add more precise formatting instructions to your prompt. In the above example, adding `"You must always return valid JSON fenced by a markdown code block. Do not return any additional text."` to your input may help steer the model to returning the expected format.
- If you are using a smaller or less capable model, try using a more capable one.
- Add [LLM-powered retries](/docs/how_to/output_parser_fixing/).
"""
logger.info("## Troubleshooting")

logger.info("\n\n[DONE]", bright=True)