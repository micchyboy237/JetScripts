from jet.logger import logger
from langchain_community.embeddings import TitanTakeoffEmbed
import os
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Titan Takeoff

`TitanML` helps businesses build and deploy better, smaller, cheaper, and faster NLP models through our training, compression, and inference optimization platform.

Our inference server, [Titan Takeoff](https://docs.titanml.co/docs/intro) enables deployment of LLMs locally on your hardware in a single command. Most embedding models are supported out of the box, if you experience trouble with a specific model, please let us know at hello@titanml.co.

## Example usage
Here are some helpful examples to get started using Titan Takeoff Server. You need to make sure Takeoff Server has been started in the background before running these commands. For more information see [docs page for launching Takeoff](https://docs.titanml.co/docs/Docs/launching/).
"""
logger.info("# Titan Takeoff")



"""
### Example 1
Basic use assuming Takeoff is running on your machine using its default ports (ie localhost:3000).
"""
logger.info("### Example 1")

embed = TitanTakeoffEmbed()
output = embed.embed_query(
    "What is the weather in London in August?", consumer_group="embed"
)
logger.debug(output)

"""
### Example 2 
Starting readers using TitanTakeoffEmbed Python Wrapper. If you haven't created any readers with first launching Takeoff, or you want to add another you can do so when you initialize the TitanTakeoffEmbed object. Just pass a list of models you want to start as the `models` parameter.

You can use `embed.query_documents` to embed multiple documents at once. The expected input is a list of strings, rather than just a string expected for the `embed_query` method.
"""
logger.info("### Example 2")

embedding_model = {
    "model_name": "BAAI/bge-large-en-v1.5",
    "device": "cpu",
    "consumer_group": "embed",
}
embed = TitanTakeoffEmbed(models=[embedding_model])

time.sleep(60)

prompt = "What is the capital of France?"
output = embed.embed_query(prompt, consumer_group="embed")
logger.debug(output)

logger.info("\n\n[DONE]", bright=True)