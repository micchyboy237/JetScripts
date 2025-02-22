from jet.libs.txtai import LLM
from typing import List, Dict
from pydantic import BaseModel
import json
from outlines.models.tokenizer import Tokenizer

# Initialize LLM
llm = LLM(path="ollama/llama3.1", method="litellm",
          api_base="http://localhost:11434")

# Define selection commands for entity types
DEFAULT_SELECTION_COMMANDS = {
    "countries": {
        "description": "Names of countries or nations",
        "examples": ["England", "France", "Spain"]
    },
    "geography": {
        "description": "Geographic features and descriptions",
        "examples": ["mountains", "coastline", "plains"]
    },
    "years": {
        "description": "Years and dates",
        "examples": ["1066", "1492", "2024"]
    },
    "people": {
        "description": "Names of people and historical figures",
        "examples": ["William", "Christopher Columbus"]
    }
}

# Define Response Model


class Response(BaseModel):
    countries: List[str] = []
    geography: List[str] = []
    years: List[str] = []
    people: List[str] = []

# Define Response Schema Validator


class JSONPrefixAllowedTokens:
    def __init__(self, schema, tokenizer, selection_commands: Dict = None, whitespace_pattern=None):
        self.schema = schema
        self.tokenizer = tokenizer
        self.selection_commands = selection_commands or DEFAULT_SELECTION_COMMANDS
        self.whitespace_pattern = whitespace_pattern or r" ?"

    def __call__(self, input_text):
        try:
            parsed = json.loads(input_text)
            self.schema(**parsed)
            return input_text
        except (json.JSONDecodeError, ValueError):
            return self._filter_tokens(input_text)

    def _filter_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        allowed_tokens = self._get_allowed_tokens()

        filtered_tokens = []
        for token in tokens:
            if token in allowed_tokens or self._is_structural_token(token):
                filtered_tokens.append(token)

        return self.tokenizer.decode(filtered_tokens)

    def _get_allowed_tokens(self):
        allowed_tokens = set()

        # Add tokens for field names and examples from selection commands
        for field, command in self.selection_commands.items():
            field_tokens = self.tokenizer.encode(field)
            allowed_tokens.update(field_tokens)

            # Add example tokens
            for example in command["examples"]:
                example_tokens = self.tokenizer.encode(example)
                allowed_tokens.update(example_tokens)

        # Add structural tokens
        structural_tokens = self.tokenizer.encode('[]",:{}')
        allowed_tokens.update(structural_tokens)

        return allowed_tokens

    def _is_structural_token(self, token):
        structural_text = self.tokenizer.decode([token])
        return structural_text in '[]",:{}' or structural_text.isspace()


def rag(question: str, text: str, prefix_allowed_tokens_fn=None) -> str:
    prompt = f"""<|im_start|>system
You are a friendly assistant that extracts and categorizes entities from text.
For each category, identify relevant items from the context:
- countries: Names of countries or nations
- geography: Geographic features and descriptions
- years: Years and dates mentioned
- people: Names of people and historical figures

Format your response as a JSON object with these categories as keys and lists as values.
<|im_end|>
<|im_start|>user
Extract and categorize entities from this context:

{text}
<|im_end|>
<|im_start|>assistant"""

    # Apply prefix_allowed_tokens_fn if provided
    if prefix_allowed_tokens_fn:
        text = prefix_allowed_tokens_fn(text)

    return llm(prompt, maxlength=4096)


def get_prefix_allowed_tokens_fn() -> JSONPrefixAllowedTokens:
    return JSONPrefixAllowedTokens(
        schema=Response,
        tokenizer=llm.generator.llm.pipeline.tokenizer,
        selection_commands=DEFAULT_SELECTION_COMMANDS,
        whitespace_pattern=r" ?"
    )


def main():
    context = """
    England's terrain chiefly consists of low hills and plains, especially in the centre and south.
    The Battle of Hastings was fought on 14 October 1066 between the Norman army of William, the Duke of Normandy, and an English army under the Anglo-Saxon King Harold Godwinson.
    Bounded by the Atlantic Ocean on the east, Brazil has a coastline of 7,491 kilometers (4,655 mi).
    Spain pioneered the exploration of the New World and the first circumnavigation of the globe.
    Christopher Columbus lands in the Caribbean in 1492.
    """

    # Guided generation with selection commands
    prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn()
    result = rag(context, context, prefix_allowed_tokens_fn)

    print("\nExtracted Entities:")
    print(json.dumps(json.loads(result), indent=2))


if __name__ == "__main__":
    main()
