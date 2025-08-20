from datasets import Dataset
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.finetuning.callbacks import MistralAIFineTuningHandler
from llama_index.finetuning.mistralai import MistralAIFinetuneEngine
from llama_index.llms.mistralai import MistralAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from tqdm import tqdm
from typing import List
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/finetuning/mistralai_fine_tuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Fine Tuning MistralAI models using Finetuning API

In this notebook, we walk through an example of fine-tuning `open-mistral-7b` using MistralAI finetuning API.

Specifically, we attempt to distill `mistral-large-latest`'s knowledge, by generating training data with `mistral-large-latest` to then fine-tune `open-mistral-7b`.

All training data is generated using two different sections of our index data, creating both a training and evalution set.

We will use `mistral-small-largest` to create synthetic training and evaluation questions to avoid any biases towards `open-mistral-7b` and `mistral-large-latest`.

We then finetune with our `MistraAIFinetuneEngine` wrapper abstraction.

Evaluation is done using the `ragas` library, which we will detail later on.

We can monitor the metrics on `Weights & Biases`
"""
logger.info("# Fine Tuning MistralAI models using Finetuning API")

# %pip install llama-index-finetuning
# %pip install llama-index-finetuning-callbacks
# %pip install llama-index-llms-mistralai
# %pip install llama-index-embeddings-mistralai



# import nest_asyncio

# nest_asyncio.apply()

"""
## Set API Key
"""
logger.info("## Set API Key")


os.environ["MISTRAL_API_KEY"] = "<YOUR MISTRALAI API KEY>"

"""
## Download Data

Here, we first down load the PDF that we will use to generate training data.
"""
logger.info("## Download Data")

# !curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf

"""
The next step is generating a training and eval dataset.

We will generate 40 training and 40 evaluation questions on different sections of the PDF we downloaded.

We can use `open-mistral-7b` on the eval questions to get our baseline performance.

Then, we will use `mistral-large-latest` on the train questions to generate our training data.

## Load Data
"""
logger.info("## Load Data")


documents = SimpleDirectoryReader(
    input_files=["IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

"""
## Setup LLM and Embedding Model
"""
logger.info("## Setup LLM and Embedding Model")


open_mistral = MistralAI(
    model="open-mistral-7b", temperature=0.1
)  # model to be finetuning
mistral_small = MistralAI(
    model="mistral-small-latest", temperature=0.1
)  # model for question generation
embed_model = MistralAIEmbedding()

"""
## Training and Evaluation Data Generation
"""
logger.info("## Training and Evaluation Data Generation")

question_gen_query = (
    "You are a Teacher/ Professor. Your task is to setup "
    "a quiz/examination. Using the provided context, formulate "
    "a single question that captures an important fact from the "
    "context. Restrict the question to the context information provided."
    "You should generate only question and nothing else."
)

dataset_generator = DatasetGenerator.from_documents(
    documents[:80],
    question_gen_query=question_gen_query,
    llm=mistral_small,
)

"""
We will generate 40 training and 40 evaluation questions
"""
logger.info("We will generate 40 training and 40 evaluation questions")

questions = dataset_generator.generate_questions_from_nodes(num=40)
logger.debug("Generated ", len(questions), " questions")

questions[10:15]

with open("train_questions.txt", "w") as f:
    for question in questions:
        f.write(question + "\n")

"""
Now, lets generate questions on a completely different set of documents, in order to create our eval dataset.
"""
logger.info("Now, lets generate questions on a completely different set of documents, in order to create our eval dataset.")

dataset_generator = DatasetGenerator.from_documents(
    documents[80:],
    question_gen_query=question_gen_query,
    llm=mistral_small,
)

questions = dataset_generator.generate_questions_from_nodes(num=40)
logger.debug("Generated ", len(questions), " questions")

with open("eval_questions.txt", "w") as f:
    for question in questions:
        f.write(question + "\n")

"""
## Initial Eval with `open-mistral-7b` Query Engine

For this eval, we will be using the [`ragas` evaluation library](https://github.com/explodinggradients/ragas).

Ragas has a ton of evaluation metrics for RAG pipelines, and you can read about them [here](https://github.com/explodinggradients/ragas/blob/main/docs/metrics.md).

For this notebook, we will be using the following two metrics

- `answer_relevancy` - This measures how relevant is the generated answer to the prompt. If the generated answer is incomplete or contains redundant information the score will be low. This is quantified by working out the chance of an LLM generating the given question using the generated answer. Values range (0,1), higher the better.
- `faithfulness` - This measures the factual consistency of the generated answer against the given context. This is done using a multi step paradigm that includes creation of statements from the generated answer followed by verifying each of these statements against the context. The answer is scaled to (0,1) range. Higher the better.
"""
logger.info("## Initial Eval with `open-mistral-7b` Query Engine")

questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())



Settings.context_window = 2048
Settings.llm = open_mistral
Settings.embed_model = MistralAIEmbedding()

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=2)

contexts = []
answers = []

for question in questions:
    response = query_engine.query(question)
    contexts.append([x.node.get_content() for x in response.source_nodes])
    answers.append(str(response))

# os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"


ds = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
)

result = evaluate(ds, [answer_relevancy, faithfulness])

"""
Let's check the results before finetuning.
"""
logger.info("Let's check the results before finetuning.")

logger.debug(result)

"""
## `mistral-large-latest` to Collect Training Data

Here, we use `mistral-large-latest` to collect data that we want `open-mistral-7b` to finetune on.
"""
logger.info("## `mistral-large-latest` to Collect Training Data")


finetuning_handler = MistralAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

llm = MistralAI(model="mistral-large-latest", temperature=0.1)
llm.callback_manager = callback_manager

questions = []
with open("train_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())


Settings.embed_model = MistralAIEmbedding()
Settings.llm = llm

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=2)


contexts = []
answers = []

for question in tqdm(questions, desc="Processing questions"):
    response = query_engine.query(question)
    contexts.append(
        "\n".join([x.node.get_content() for x in response.source_nodes])
    )
    answers.append(str(response))



def convert_data_jsonl_format(
    questions: List[str],
    contexts: List[str],
    answers: List[str],
    output_file: str,
) -> None:
    with open(output_file, "w") as outfile:
        for context, question, answer in zip(contexts, questions, answers):
            message_dict = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "You are a helpful assistant to answer user queries based on provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"context: {context} \n\n question: {question}",
                    },
                    {"role": "assistant", "content": answer},
                ]
            }
            json.dump(message_dict, outfile)
            outfile.write("\n")

convert_data_jsonl_format(questions, contexts, answers, "training.jsonl")

"""
## Create `MistralAIFinetuneEngine`

We create an `MistralAIFinetuneEngine`: the finetune engine will take care of launching a finetuning job, and returning an LLM model that you can directly plugin to the rest of LlamaIndex workflows.

We use the default constructor, but we can also directly pass in our finetuning_handler into this engine with the `from_finetuning_handler` class method.
"""
logger.info("## Create `MistralAIFinetuneEngine`")


wandb_integration_dict = {
    "project": "mistralai",
    "run_name": "finetuning",
    "api_key": "<api_key>",
}

finetuning_engine = MistralAIFinetuneEngine(
    base_model="open-mistral-7b",
    training_path="training.jsonl",
    verbose=True,
    training_steps=5,
    learning_rate=0.0001,
    wandb_integration_dict=wandb_integration_dict,
)

finetuning_engine.finetune()

finetuning_engine.get_current_job()

finetuning_engine.get_current_job()

ft_llm = finetuning_engine.get_finetuned_model(temperature=0.1)

"""
## Evaluation

Once the finetuned model is created, the next step is running our fine-tuned model on our eval dataset again to measure any performance increase.
"""
logger.info("## Evaluation")


Settings.llm = ft_llm
Settings.context_window = (
    2048  # limit the context window artifically to test refine process
)
Settings.embed_model = MistralAIEmbedding()

questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())


index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=2, llm=ft_llm)

contexts = []
answers = []

for question in tqdm(questions, desc="Processing Questions"):
    response = query_engine.query(question)
    contexts.append([x.node.get_content() for x in response.source_nodes])
    answers.append(str(response))

ds = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
)

result = evaluate(ds, [answer_relevancy, faithfulness])

"""
Let's check the results with finetuned model
"""
logger.info("Let's check the results with finetuned model")

logger.debug(result)

"""
## Observation:

`open-mistral-7b` : 'answer_relevancy': **0.8151**, 'faithfulness': **0.8360**

`open-mistral-7b-finetuned` : 'answer_relevancy': **0.8016**, 'faithfulness': **0.8924**

As you can see there is an increase in faithfulness score and small drop in answer relevancy.

## Exploring Differences

Let's quickly compare the differences in responses, to demonstrate that fine tuning did indeed change something.
"""
logger.info("## Observation:")


index = VectorStoreIndex.from_documents(documents)

questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())

logger.debug(questions[20])

"""
### Original `open-mistral-7b  `
"""
logger.info("### Original `open-mistral-7b  `")


query_engine = index.as_query_engine(llm=open_mistral)

response = query_engine.query(questions[20])

display_response(response)

"""
### Fine-Tuned `open-mistral-7b`
"""
logger.info("### Fine-Tuned `open-mistral-7b`")

query_engine = index.as_query_engine(llm=ft_llm)

response = query_engine.query(questions[20])

display_response(response)

"""
As we can see, the fine-tuned model provides a more thorough response! This lines up with the increased faithfullness score from ragas, since the answer is more representative of the retrieved context.

## Conclusion

So, in conclusion, finetuning with only ~40 questions actually helped improve our eval scores!

**answer_relevancy: 0.0.8151 -> 0.8016**

The answer relevancy dips slightly but it's very small.

**faithfulness: 0.8360 -> 0.8924**

The faithfulness appears to have been improved! This mains the anwers given better fuffil the original question that was asked.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)