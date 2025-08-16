

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Cookbook

This section contains a collection of recipes that demonstrate how to use the Core API features.

## List of recipes
"""
logger.info("# Cookbook")

:maxdepth: 1

azure-openai-with-aad-auth
termination-with-intervention
tool-use-with-intervention
extracting-results-with-an-agent
openai-assistant-agent
langgraph-agent
llamaindex-agent
local-llms-ollama-litellm
instrumenting
topic-subscription-scenarios
structured-output-agent
llm-usage-logger

logger.info("\n\n[DONE]", bright=True)