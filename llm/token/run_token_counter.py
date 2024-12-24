from jet.llm.token import token_counter
from jet.llm.llm_types import Message
from jet.llm.ollama import OLLAMA_HF_MODELS, count_tokens

texts = [
    "This is the first example sentence.",
    "Here is another text to test the function.",
    "Let's see how the token count works with multiple items in the batch.",
    "This is the last sentence in the example batch."
]

# Model key from OLLAMA_HF_MODELS
model = "llama3.1"
# model = "mistral"
num_tokens = token_counter(texts, model=model)
print(f"Token count: {num_tokens}")
