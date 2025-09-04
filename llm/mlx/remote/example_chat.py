from jet.llm.mlx.remote import generation as gen


def main():
    print("=== Chat Completion ===")
    response = gen.chat(
        "Write a haiku about the ocean.",
        model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    )
    print(response)


if __name__ == "__main__":
    main()
