from jet.helpers.prompt.custom_prompt_templates import MetadataPromptTemplate
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms.llm import LLM
from jet.logger import logger


def main():
    # Define a sample LLM metadata (for example purposes)
    llm_metadata = {
        "context_window": 2048,  # Example context window size
        "num_output": 200,  # Example number of output tokens reserved
    }

    # Create a PromptHelper instance
    prompt_helper = PromptHelper.from_llm_metadata(llm_metadata)

    # Define a prompt template
    template = "Please summarize the following text: {text}"

    # Create a MetadataPromptTemplate with the defined template
    prompt_template = MetadataPromptTemplate(
        template=template,
        prompt_type="CUSTOM"
    )

    # Define text chunks to be processed
    text_chunks = [
        "This is some text to be summarized. It is long enough to test chunking."]

    # Use the PromptHelper to truncate or repack text chunks
    truncated_chunks = prompt_helper.truncate(
        prompt_template, text_chunks, padding=10
    )

    # Use the PromptHelper to repack text chunks if needed
    repacked_chunks = prompt_helper.repack(
        prompt_template, text_chunks, padding=10
    )

    # Format the prompt template with specific text data
    formatted_prompt = prompt_template.format(text="This is an example text.")

    logger(f"Formatted Prompt: {formatted_prompt}")
    logger(f"Truncated Chunks: {truncated_chunks}")
    logger(f"Repacked Chunks: {repacked_chunks}")


if __name__ == "__main__":
    main()
