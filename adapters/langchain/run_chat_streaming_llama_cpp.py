from langchain_openai import ChatOpenAI
from jet.logger import logger

llm = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    streaming=True,  # Enables token-by-token streaming
)

def stream_llm_response(llm: ChatOpenAI, prompt: str) -> str:
    """
    Streams the LLM response, printing and logging chunks in real-time.
    
    Args:
        llm: Initialized ChatOpenAI instance.
        prompt: Input string (or adapt for messages list).
    
    Returns:
        Full concatenated response.
    """
    full_response = ""
    logger.info("Starting LLM stream for prompt: %s", prompt)
    
    # Stream chunks (each is an AIMessageChunk)
    for chunk in llm.stream(prompt):
        content = chunk.content
        if content:  # Skip empty chunks
            logger.teal(content, flush=True)  # Print chunk immediately
            logger.info("Stream chunk: %s", content)  # Log each chunk
            full_response += content
    
    print()  # Newline after full response
    logger.info("Completed LLM stream. Full response length: %d", len(full_response))
    return full_response

# Example invocation
response = stream_llm_response(llm, "Explain quantum computing in simple terms.")