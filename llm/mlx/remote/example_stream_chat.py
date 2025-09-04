from jet.llm.mlx.remote import generation as gen


def main():
    print("=== Streaming Chat Completion ===")
    for chunk in gen.stream_chat(
        "Explain the benefits of unit testing in Python.",
        model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    ):
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                print(delta, end="", flush=True)
    print("\n--- Stream End ---")


if __name__ == "__main__":
    main()
