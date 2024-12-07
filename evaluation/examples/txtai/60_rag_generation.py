from txtai import LLM
from typing import List
from pydantic import BaseModel
import json
from outlines.models.tokenizer import Tokenizer

# Initialize LLM
llm = LLM(path="ollama/llama3.1", method="litellm",
          api_base="http://localhost:11434")


# Define Response Model
class Response(BaseModel):
    countries: List[str] = []
    geography: List[str] = []
    years: List[str] = []
    people: List[str] = []


# Define Response Schema Validator
class JSONPrefixAllowedTokens:
    def __init__(self, schema, tokenizer_or_pipe, whitespace_pattern=None):
        self.schema = schema
        self.tokenizer = tokenizer_or_pipe
        self.whitespace_pattern = whitespace_pattern or r" ?"

    def __call__(self, text):
        # Tokenize the text using the tokenizer
        tokens = self.tokenizer.encode(text)

        # Ensure only valid tokens are used (according to the schema)
        allowed_tokens = self.get_allowed_tokens()

        # Filter tokens by allowed ones based on the schema
        filtered_tokens = [
            token for token in tokens if token in allowed_tokens]

        # Reconstruct the text with allowed tokens
        decoded_text = self.tokenizer.decode(filtered_tokens)

        return decoded_text

    def get_allowed_tokens(self):
        # Create a list of allowed tokens based on the schema
        allowed_tokens = set()

        # Iterate through the schema and add the allowed tokens for each field
        for field in self.schema.__annotations__:
            allowed_tokens.add(self.schema.__annotations__[field])

        return allowed_tokens


# Define the RAG process
def rag(question: str, text: str, prefix_allowed_tokens_fn=None) -> str:
    prompt = f"""<|im_start|>system
    You are a friendly assistant. You answer questions from users.<|im_end|>
    <|im_start|>user
    Answer the following question using only the context below. Only include information specifically discussed.

    question: {question}
    context: {text} <|im_end|>
    <|im_start|>assistant
    """
    # Apply prefix_allowed_tokens_fn if provided
    if prefix_allowed_tokens_fn:
        text = prefix_allowed_tokens_fn(text)

    return llm(prompt, maxlength=4096)


# Define prefix_allowed_tokens_fn for guided generation
# This method applies a custom handler to constrain/guide how the LLM generates tokens.
def get_prefix_allowed_tokens_fn() -> JSONPrefixAllowedTokens:
    return JSONPrefixAllowedTokens(
        schema=Response,
        tokenizer_or_pipe=llm.generator.llm.pipeline.tokenizer,
        whitespace_pattern=r" ?"
    )


# Main function
def main():
    # Manually generated context. Replace with an Embedding search or other request.
    context = """
    England's terrain chiefly consists of low hills and plains, especially in the centre and south.
    The Battle of Hastings was fought on 14 October 1066 between the Norman army of William, the Duke of Normandy, and an English army under the Anglo-Saxon King Harold Godwinson.
    Bounded by the Atlantic Ocean on the east, Brazil has a coastline of 7,491 kilometers (4,655 mi).
    Spain pioneered the exploration of the New World and the first circumnavigation of the globe.
    Christopher Columbus lands in the Caribbean in 1492.
    """

    # Call rag function without any additional constraints
    print("Standard RAG Output:")
    print(rag("List the countries discussed", context))

    # Guided generation
    prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn()
    print("\nGuided Generation Output:")
    print(json.loads(rag("List the entities discussed",
          context, prefix_allowed_tokens_fn)))


if __name__ == "__main__":
    main()
