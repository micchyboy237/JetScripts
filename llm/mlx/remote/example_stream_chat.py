from jet.llm.mlx.remote import generation as gen


def main():
    print("=== Streaming Chat Completion ===")
    for chunk in gen.stream_chat(
        "Explain the benefits of unit testing in Python.",
        model=None,
        max_tokens=100
    ):
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0]["message"]["content"]
            if delta:
                print(delta, end="", flush=True)
    print("\n--- Stream End ---")


if __name__ == "__main__":
    main()
