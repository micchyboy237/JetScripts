from jet.logger import logger


def generate_export(arg):
    return f"export SAMPLE_ARG='{arg}'"


# Output only the final command
if __name__ == "__main__":
    sample_arg = "HelloWorld"
    logger.debug("Calling generate_export")
    print(generate_export(sample_arg))
