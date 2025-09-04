from jet.llm.mlx.remote import generation as gen


def main():
    print("=== Streaming Text Generation ===")
    for chunk in gen.stream_generate(
        "In the future, AI assistants will",
        model="mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_tokens=50,
    ):
        if "choices" in chunk and chunk["choices"]:
            token = chunk["choices"][0].get("text")
            if token:
                print(token, end="", flush=True)
    print("\n--- Stream End ---")


if __name__ == "__main__":
    main()
