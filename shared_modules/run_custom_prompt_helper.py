import logging
from typing import Optional, Sequence
from jet.helpers.prompt.custom_prompt_helpers import OllamaPromptHelper
from jet.token.token_utils import get_tokenizer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.prompts import BasePromptTemplate, ChatPromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.tools import BaseTool
from llama_index.core.node_parser.text.token import TokenTextSplitter
from llama_index.core.indices.prompt_helper import PromptHelper
from jet.llm.ollama.base import Ollama, initialize_ollama_settings
from jet.logger import logger
llm_settings = initialize_ollama_settings()


# Example Usage
def main():
    # Create a sample prompt template (ChatPromptTemplate)
    prompt = ChatPromptTemplate(
        message_templates=[ChatMessage(
            content="What is the weather like today?")],
    )

    # Initialize OllamaPromptHelper
    helper = OllamaPromptHelper(
        llm=llm_settings.llm
    )

    # Sample text chunks to work with
    text_chunks = ["The weather is sunny and warm.",
                   "A great day to go outside!", "Temperature around 75°F."]

    # Truncate text chunks to fit within context size
    truncated_text = helper.truncate(prompt, text_chunks)
    logger.info("Truncated Text:")
    logger.success(truncated_text)

    # Repack text chunks to maximize the context window
    repacked_text = helper.repack(prompt, text_chunks)
    logger.info("Repacked Text:")
    logger.success(repacked_text)


if __name__ == "__main__":
    main()
