from jet.logger import logger
from langchain_community.llms import Banana
import os
import shutil


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
# Banana

>[Banana](https://www.banana.dev/) provided serverless GPU inference for AI models,
> a CI/CD build pipeline and a simple Python framework (`Potassium`) to server your models.

This page covers how to use the [Banana](https://www.banana.dev) ecosystem within LangChain.

## Installation and Setup

- Install the python package `banana-dev`:
"""
logger.info("# Banana")

pip install banana-dev

"""
- Get an Banana api key from the [Banana.dev dashboard](https://app.banana.dev) and set it as an environment variable (`BANANA_API_KEY`)
- Get your model's key and url slug from the model's details page.

## Define your Banana Template

You'll need to set up a Github repo for your Banana app. You can get started in 5 minutes using [this guide](https://docs.banana.dev/banana-docs/).

Alternatively, for a ready-to-go LLM example, you can check out Banana's [CodeLlama-7B-Instruct-GPTQ](https://github.com/bananaml/demo-codellama-7b-instruct-gptq) GitHub repository. Just fork it and deploy it within Banana.

Other starter repos are available [here](https://github.com/orgs/bananaml/repositories?q=demo-&type=all&language=&sort=).

## Build the Banana app

To use Banana apps within Langchain, you must include the `outputs` key
in the returned json, and the value must be a string.
"""
logger.info("## Define your Banana Template")

result = {'outputs': result}

"""
An example inference function would be:
"""
logger.info("An example inference function would be:")

@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    """Handle a request to generate code from a prompt."""
    model = context.get("model")
    tokenizer = context.get("tokenizer")
    max_new_tokens = request.json.get("max_new_tokens", 512)
    temperature = request.json.get("temperature", 0.7)
    prompt = request.json.get("prompt")
    prompt_template=f'''[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
    {prompt}
    [/INST]
    '''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(output[0])
    return Response(json={"outputs": result}, status=200)

"""
This example is from the `app.py` file in [CodeLlama-7B-Instruct-GPTQ](https://github.com/bananaml/demo-codellama-7b-instruct-gptq).


## LLM
"""
logger.info("## LLM")


"""
See a [usage example](/docs/integrations/llms/banana).
"""
logger.info("See a [usage example](/docs/integrations/llms/banana).")

logger.info("\n\n[DONE]", bright=True)