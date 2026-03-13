import os

import dspy

# For any OpenAI-compatible endpoint (vLLM, LM Studio, Ollama, LocalAI, OpenRouter, Together, Groq, Fireworks, Azure with custom endpoint, etc.)
lm = dspy.LM(
    model="openai/ministral-3-3b-instruct",  # ← prefix with "openai/" when using a custom base
    api_base=os.getenv("LLAMA_CPP_LLM_URL"),  # ← your custom base URL here
    # api_base="https://api.your-provider.com/v1",
    api_key="sk-1234",
    model_type="chat",  # usually "chat" for modern endpoints
    temperature=0.7,
    max_tokens=1024,
    cache=False,  # optional: useful during dev
    # other kwargs go to LiteLLM / the endpoint
)

# Then set it globally
dspy.configure(lm=lm)


# Define a signature (input → output spec)
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")


# Create a module (Predict is the simplest)
generate_answer = dspy.Predict(BasicQA)

# Run it
pred = generate_answer(question="What is the capital of Japan?")
print(pred.answer)  # e.g. "Tokyo"
