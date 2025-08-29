from PIL import Image
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.openai import OllamaFunctionCallingAdapterMultiModal
import matplotlib.pyplot as plt
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/gpt4v_experiments_cot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# GPT4-V Experiments with General, Specific questions and Chain Of Thought (COT) Prompting Technique.

GPT-4V has amazed us with its ability to analyze images and even generate website code from visuals.

This tutorial notebook investigates GPT-4V's proficiency in interpreting bar charts, scatter plots, and tables. We aim to assess whether specific questioning and chain of thought prompting can yield better responses compared to broader inquiries. Our demonstration seeks to determine if GPT-4V can exceed these known limitations with precise questioning and systematic reasoning techniques.

We observed in these experiments that asking specific questions, rather than general ones, yields better answers. Let's delve into these experiments.

NOTE: This tutorial notebook aims to inform the community about GPT-4V's performance, though the results might not be universally applicable. We strongly advise conducting tests with similar questions on your own dataset before drawing conclusions.

We have put to test following images from [Llama2](https://arxiv.org/pdf/2307.09288.pdf) and [MistralAI](https://arxiv.org/pdf/2310.06825.pdf) papers.

1. Violation percentage of safety with different LLMs across categories. (Llama2 paper)
2. Llama2 vs Mistral model performances across various NLP tasks.(Mistral paper)
2. Performances of different LLMs across various NLP tasks. (Llama2 paper)

Let's inspect each of these images now.

Let's start analyzing these images by following these steps for our questions:

1. General Question: Simply ask, "Analyze the image."
2. Specific Inquiry: Question the performance of a certain category or model in detail.
3. Chain of Thought Prompting: Use a step-by-step reasoning method to walk through the analysis.

These guidelines aim to test how different questioning techniques might improve the precision of the information we gather from the images.
"""
logger.info("# GPT4-V Experiments with General, Specific questions and Chain Of Thought (COT) Prompting Technique.")

# %pip install llama-index-multi-modal-llms-openai

# !pip install llama-index


# OPENAI_API_KEY = "YOUR OPENAI API KEY"

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


openai_mm_llm = OllamaFunctionCallingAdapterMultiModal(
    model="llama3.2", request_timeout=300.0, context_window=4096,
#     api_key=OPENAI_API_KEY,
    max_new_tokens=500,
    temperature=0.0,
)

"""
### Download Data
"""
logger.info("### Download Data")

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/gpt4_experiments/llama2_mistral.png' -O './llama2_mistral.png'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/gpt4_experiments/llama2_model_analysis.pdf' -O './llama2_model_analysis.png'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/gpt4_experiments/llama2_violations_charts.png' -O './llama2_violations_charts.png'

"""
### Image1 - Violation percentage of safety with different LLMs across categories.
"""
logger.info("### Image1 - Violation percentage of safety with different LLMs across categories.")


img = Image.open("llama2_violations_charts.png")
plt.imshow(img)

image_documents = SimpleDirectoryReader(
    input_files=["./llama2_violations_charts.png"]
).load_data()

"""
#### General Question
"""
logger.info("#### General Question")

query = "Analyse the image"

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

As you can see though the categories hateful and harmful, illicit and criminal activity, and unqualified advice but it hallicunated with x-axis values with -  "Video sharing", "Social networking", "Gaming", "Dating", "Forums & boards", "Commercial Websites", "Media sharing", "P2P/File sharing", "Wiki", and "Other".

#### Specific Questions
"""
logger.info("#### Observation:")

query = "Compare Llama2 models vs Vicuna models across categories."

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

It answered wrong by saying Vicuna model generally has a lower violation percentage across all subcategories compared to the Llama2 model.
"""
logger.info("#### Observation:")

query = "which model among llama2 and vicuna models does better in terms of violation percentages in Hateful and harmful category."

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

It failed to accurately capture the information, mistakenly identifying the light blue bar as representing Vicuna when, in fact, it is the light blue bar that represents Llama2.

Now let's inspect by giving more detailed information and ask the same question.
"""
logger.info("#### Observation:")

query = """In the image provided to you depicts about the violation rate performance of various AI models across Hateful and harmful, Illicit and criminal activity, Unqualified advice categories.
           Hateful and harmful category is in first column. Bars with light blue are with Llama2 model and dark blue are with Vicuna models.
           With this information, Can you compare about Llama2 and Vicuna models in Hateful and harmful category."""

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

It did answer the question correctly.

#### Chain of thought prompting
"""
logger.info("#### Observation:")

query = """Based on the image provided. Follow the steps and answer the query - which model among llama2 and vicuna does better in terms of violation percentages in 'Hateful and harmful'.

Examine the Image: Look at the mentioned category in the query in the Image.

Identify Relevant Data: Note the violation percentages.

Evaluate: Compare if there is any comparison required as per the query.

Draw a Conclusion: Now draw the conclusion based on the whole data."""

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

With chain of thought prompting it did hallicunate with bar colours but answered correctly saying Llama2 has lower violation compared to vicuna in Hateful and harmful though for a section Llama2 has higher violation compared to vicuna.

### Image2 - Llama2 vs Mistral model performances across various NLP tasks.
"""
logger.info("#### Observation:")

img = Image.open("llama2_mistral.png")
plt.imshow(img)

image_documents = SimpleDirectoryReader(
    input_files=["./llama2_mistral.png"]
).load_data()

"""
#### General Question
"""
logger.info("#### General Question")

query = "Analyse the image"

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:
It did answer the query but hallicunated with NLU task which is MMLU task and assumed Mistral is available across all different model parameters.

#### Specific Questions
"""
logger.info("#### Observation:")

query = "How well does mistral model compared to llama2 model?"

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:
Incorrect answer and percentages are not accurate enough and again assumed mistral is available across all parameter models.
"""
logger.info("#### Observation:")

query = "Assuming mistral is available in 7B series. How well does mistral model compared to llama2 model?"

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:
Now with giving the detail that mistral is available in 7B series, it is able to answer correctly.

### Chain of thought prompting.
"""
logger.info("#### Observation:")

query = """Based on the image provided. Follow the steps and answer the query - Assuming mistral is available in 7B series. How well does mistral model compared to llama2 model?.

Examine the Image: Look at the mentioned category in the query in the Image.

Identify Relevant Data: Note the respective percentages.

Evaluate: Compare if there is any comparison required as per the query.

Draw a Conclusion: Now draw the conclusion based on the whole data."""

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

There is hallicunation with number of model parameters and percentage points though the final conclusion is partially correct.

### Image3 - Performances of different LLMs across various NLP tasks.
"""
logger.info("#### Observation:")

img = Image.open("llm_analysis.png")
plt.imshow(img)

image_documents = SimpleDirectoryReader(
    input_files=["./llama2_model_analysis.png"]
).load_data()

"""
#### General Question
"""
logger.info("#### General Question")

query = "Analyse the image"

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

It did not analyse the image specifically but understood the overall data present in the image to some extent.

#### Specific Questions
"""
logger.info("#### Observation:")

query = "which model has higher performance in SAT-en?"

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

It did answer correctly but the numbers are being hallicunated.
"""
logger.info("#### Observation:")

query = "which model has higher performance in SAT-en in 7B series models?"

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

It did pick up the model names and answered correctly but recognised Llama series of models and values incorrectly.

### Chain of thought prompting.
"""
logger.info("#### Observation:")

query = """Based on the image provided. Follow the steps and answer the query - which model has higher performance in SAT-en in 7B series models?

Examine the Image: Look at the mentioned category in the query in the Image.

Identify Relevant Data: Note the respective percentages.

Evaluate: Compare if there is any comparison required as per the query.

Draw a Conclusion: Now draw the conclusion based on the whole data."""

response_gpt4v = openai_mm_llm.complete(
    prompt=query,
    image_documents=image_documents,
)

logger.debug(response_gpt4v)

"""
#### Observation:

With chain of the thought prompting we are able to get right conclusion though it should be noted that it picked up wrong values.

## Final Observations:
Observations made based on experiments on Hallucination and correctness. 

(Please note that these observations are specific to the images used and cannot be generalized, as they vary depending on the images.)

![image.png](attachment:image.png)

### Summary

In this tutorial notebook, we have showcased experiments ranging from general inquiries to systematic questions and chain of thought prompting techniques and observed Hallucination and correctness metrics.

However, it should be noted that the outputs from GPT-4V can be somewhat inconsistent, and the levels of hallucination are slightly elevated. Therefore, repeating the same experiment could result in different answers, particularly with generalized questions.
"""
logger.info("#### Observation:")

logger.info("\n\n[DONE]", bright=True)