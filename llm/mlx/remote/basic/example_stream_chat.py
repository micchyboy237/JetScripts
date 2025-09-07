from jet.llm.mlx.remote import generation as gen


def main():
    print("=== Streaming Chat Completion ===")
    response = ""
    for chunk in gen.stream_chat(
        "Explain all the benefits of unit testing in Python.",
        verbose=True
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
    print("\n--- Stream End ---")


if __name__ == "__main__":
    main()
