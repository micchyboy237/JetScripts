from jet.llm.mlx.remote import generation as gen


def main():
    print("=== Available Models ===")
    models = gen.get_models()
    print(models)


if __name__ == "__main__":
    main()
