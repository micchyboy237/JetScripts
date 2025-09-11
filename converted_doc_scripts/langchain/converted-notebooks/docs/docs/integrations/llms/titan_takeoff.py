from jet.logger import logger
from langchain_community.llms import TitanTakeoff
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
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

Our inference server, [Titan Takeoff](https://docs.titanml.co/docs/intro) enables deployment of LLMs locally on your hardware in a single command. Most generative model architectures are supported, such as Falcon, Llama 2, GPT2, T5 and many more. If you experience trouble with a specific model, please let us know at hello@titanml.co.

## Example usage
Here are some helpful examples to get started using Titan Takeoff Server. You need to make sure Takeoff Server has been started in the background before running these commands. For more information see [docs page for launching Takeoff](https://docs.titanml.co/docs/Docs/launching/).
"""
logger.info("# Titan Takeoff")



"""
### Example 1

Basic use assuming Takeoff is running on your machine using its default ports (ie localhost:3000).
"""
logger.info("### Example 1")

llm = TitanTakeoff()
output = llm.invoke("What is the weather in London in August?")
logger.debug(output)

"""
### Example 2

Specifying a port and other generation parameters
"""
logger.info("### Example 2")

llm = TitanTakeoff(port=3000)
output = llm.invoke(
    "What is the largest rainforest in the world?",
    consumer_group="primary",
    min_new_tokens=128,
    max_new_tokens=512,
    no_repeat_ngram_size=2,
    sampling_topk=1,
    sampling_topp=1.0,
    sampling_temperature=1.0,
    repetition_penalty=1.0,
    regex_string="",
    json_schema=None,
)
logger.debug(output)

"""
### Example 3

Using generate for multiple inputs
"""
logger.info("### Example 3")

llm = TitanTakeoff()
rich_output = llm.generate(["What is Deep Learning?", "What is Machine Learning?"])
logger.debug(rich_output.generations)

"""
### Example 4

Streaming output
"""
logger.info("### Example 4")

llm = TitanTakeoff(
    streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
prompt = "What is the capital of France?"
output = llm.invoke(prompt)
logger.debug(output)

"""
### Example 5

Using LCEL
"""
logger.info("### Example 5")

llm = TitanTakeoff()
prompt = PromptTemplate.from_template("Tell me about {topic}")
chain = prompt | llm
output = chain.invoke({"topic": "the universe"})
logger.debug(output)

"""
### Example 6

Starting readers using TitanTakeoff Python Wrapper. If you haven't created any readers with first launching Takeoff, or you want to add another you can do so when you initialize the TitanTakeoff object. Just pass a list of model configs you want to start as the `models` parameter.
"""
logger.info("### Example 6")

llama_model = {
    "model_name": "TheBloke/Llama-2-7b-Chat-AWQ",
    "device": "cuda",
    "consumer_group": "llama",
}
llm = TitanTakeoff(models=[llama_model])

time.sleep(60)

prompt = "What is the capital of France?"
output = llm.invoke(prompt, consumer_group="llama")
logger.debug(output)

logger.info("\n\n[DONE]", bright=True)