from jet.llm.mlx.remote import generation as gen


def main():
    print("=== Health Check ===")
    status = gen.health_check()
    print(status)


if __name__ == "__main__":
    main()
