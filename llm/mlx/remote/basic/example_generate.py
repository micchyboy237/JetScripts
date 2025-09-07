from jet.llm.mlx.remote import generation as gen


def main():
    print("=== Text Generation ===")
    response = gen.generate(
        "Once upon a time in a faraway land,",
        model="mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_tokens=50,
        temperature=0.7,
    )
    print(response)


if __name__ == "__main__":
    main()
