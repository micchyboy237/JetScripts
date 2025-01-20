from jet.llm.ollama.base import initialize_ollama_settings
from jet.logger import logger
from llama_index.core import QueryBundle
from llama_index.core.tools import ToolMetadata
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.question_gen.openai import OpenAIQuestionGenerator
from jet.llm.ollama import Ollama
initialize_ollama_settings()

llm = Ollama(model="mistral")
question_gen = OpenAIQuestionGenerator.from_defaults(llm=llm)

logger.newline()
logger.info("Generated Prompts:")
logger.success(format_json(question_gen.get_prompts()))


tool_choices = [
    ToolMetadata(
        name="uber_2021_10k",
        description=(
            "Provides information about Uber financials for year 2021"
        ),
    ),
    ToolMetadata(
        name="lyft_2021_10k",
        description=(
            "Provides information about Lyft financials for year 2021"
        ),
    ),
]


query_str = "Compare and contrast Uber and Lyft"
choices = question_gen.generate(tool_choices, QueryBundle(query_str=query_str))

logger.newline()
logger.info("Chpices:")
logger.success(format_json(choices))
