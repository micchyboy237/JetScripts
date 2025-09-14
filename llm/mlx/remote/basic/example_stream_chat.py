from jet.llm.mlx.remote import generation as gen


def main():
    print("=== Streaming Chat Completion ===")
    response = ""
    for chunk in gen.stream_chat(
        "Explain all the benefits of unit testing in Python.",
        model="llama-3.2-3b-instruct-4bit",
        temperature=0.7,
        verbose=True,
        max_tokens=50,
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
    print("\n--- Stream End ---")


if __name__ == "__main__":
    main()
