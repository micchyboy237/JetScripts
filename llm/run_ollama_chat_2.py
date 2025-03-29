from jet.actions.generation import call_ollama_chat
from jet.logger import logger

# Main function to demonstrate sample usage
if __name__ == "__main__":
    prompt = "Write a 20 word creative story about an explorer finding a hidden treasure."

    # generator = Ollama()
    # result = generator.generate(prompt)
    # print("Generated Output:")
    # print(result["output"])

    response = ""
    for chunk in call_ollama_chat(
        prompt,
        track={"repo": "./test", "run_name": "Short story"}
    ):
        response += chunk
        logger.success(chunk, flush=True)
