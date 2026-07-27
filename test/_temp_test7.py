import json
import os
from typing import Type, TypeVar

from jet.adapters.llama_cpp.factory import get_llm_client
from jet.logger import logger
from openai import Stream
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, ValidationError

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


def extract_entities_from_text(
    text: str,
    model_class: Type[T],
    temperature: float = 0.2,
    max_tokens: int = 1000,
    timeout: float = 30.0,
) -> T:
    """
    Extract structured entities from text using a local llama.cpp LLM server.

    This function uses OpenAI-compatible API to call a local llama.cpp server
    and extract entities based on a Pydantic model schema.

    Args:
        text: The input text from which to extract entities.
        model_class: A Pydantic BaseModel class defining the structure of entities to extract.
        temperature: Sampling temperature (0-2). Lower values make output more deterministic.
        max_tokens: Maximum number of tokens to generate in the response.
        timeout: Request timeout in seconds.

    Returns:
        An instance of the provided Pydantic model with extracted entities.

    Raises:
        ValueError: If the LLM response cannot be parsed into the expected model.
        ValidationError: If the extracted data doesn't match the Pydantic model schema.
        Exception: If there's an error communicating with the LLM server.

    Example:
        >>> from pydantic import BaseModel, Field
        >>>
        >>> class Person(BaseModel):
        ...     name: str = Field(description="Person's full name")
        ...     age: int = Field(description="Person's age in years")
        ...     occupation: str = Field(description="Person's job or profession")
        >>>
        >>> text = "John Doe is a 35-year-old software engineer."
        >>> result = extract_entities_from_text(text, Person)
        >>> print(result.name)  # "John Doe"
        >>> print(result.age)   # 35
    """

    # Get configuration from environment variables
    model_name = os.getenv("LLAMA_CPP_LLM_MODEL", "qwen3.5-uncensored:2b")
    host = os.getenv("LLAMA_CPP_LLM_HOST", "http://localhost:8080")

    # Initialize OpenAI client pointing to local llama.cpp server
    client = get_llm_client()

    # Convert Pydantic model to JSON schema
    schema = model_class.model_json_schema()

    # Create system prompt with instructions
    system_prompt = f"""You are an entity extraction assistant. Your task is to extract structured information from the given text according to the provided JSON schema.

Instructions:
1. Read the input text carefully
2. Extract only the information that matches the schema
3. Return ONLY valid JSON matching the schema exactly
4. Do not include any explanations, markdown formatting, or additional text
5. If information is not available in the text, use null for optional fields or best guess for required fields

JSON Schema:
{json.dumps(schema, indent=2)}
"""

    # Create user message with the text to process
    user_message = f"""Extract entities from the following text:

{text}"""

    try:
        # Call the LLM using OpenAI-compatible API with structured output
        stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            stream=True,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )

        # Extract the response content
        content = ""
        for part in stream:
            if part.choices and part.choices[0].delta:
                delta = part.choices[0].delta

                # Check for reasoning_content first
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    content += delta.reasoning_content
                    logger.orange(delta.reasoning_content, flush=True, end="")

                # Then check for regular content
                elif hasattr(delta, "content") and delta.content:
                    content += delta.content
                    logger.teal(delta.content, flush=True, end="")

        if not content:
            raise ValueError("Empty response from LLM")

        # Parse the JSON response
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response if it contains extra text
            import re

            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL
            )
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError(f"Failed to parse JSON from LLM response: {content}")

        # Validate and return the Pydantic model instance
        return model_class(**data)

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise Exception(f"Error extracting entities: {str(e)}") from e


if __name__ == "__main__":
    # Example usage
    from typing import Optional

    from pydantic import BaseModel, Field

    class PersonInfo(BaseModel):
        name: str = Field(description="Person's full name")
        age: Optional[int] = Field(default=None, description="Person's age in years")
        occupation: Optional[str] = Field(
            default=None, description="Person's job or profession"
        )
        location: Optional[str] = Field(
            default=None, description="City or location where person lives"
        )

    sample_text = """
    Sarah Johnson is a 28-year-old data scientist living in San Francisco. 
    She specializes in machine learning and natural language processing.
    """

    try:
        result = extract_entities_from_text(sample_text, PersonInfo)
        print("Extracted entities:")
        print(f"  Name: {result.name}")
        print(f"  Age: {result.age}")
        print(f"  Occupation: {result.occupation}")
        print(f"  Location: {result.location}")
    except Exception as e:
        print(f"Error: {e}")
