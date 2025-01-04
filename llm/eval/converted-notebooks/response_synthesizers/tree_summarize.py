
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import SimpleDirectoryReader
from jet.server import predict_entities
from jet.transformers import make_serializable
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings

import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/response_synthesizers/tree_summarize.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Tree Summarize

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# !pip install llama-index

# Download Data

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# Load Data


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume").load_data()
logger.log("Documents:", len(documents), colors=["WHITE", "DEBUG"])

texts = [doc.text for doc in documents]
logger.success(json.dumps(texts, indent=2))


# Predict labels
logger.newline()
logger.newline()
logger.debug("Predicting labels...")
for text in texts:
    entities = predict_entities(
        text=text,
        model_name="tomaarsen/span-marker-mbert-base-multinerd",
        # model_name="tomaarsen/span-marker-roberta-large-ontonotes5",
    )
    entities = make_serializable(entities)

    logger.newline()
    logger.info(text)

    for entity in entities:
        logger.newline()
        logger.log("Text:", entity['span'], colors=["WHITE", "INFO"])
        logger.log("Label:", entity['label'], colors=["WHITE", "SUCCESS"])
        logger.log("Confidence:", f"{entity['score']:.4f}", colors=[
            "WHITE", "SUCCESS"])
        logger.log("Start:", f"{entity['char_start_index']}", colors=[
            "WHITE", "SUCCESS"])
        logger.log("End:", f"{entity['char_end_index']}",
                   colors=["WHITE", "SUCCESS"])
        logger.log("---")


# Summarize
logger.newline()
logger.newline()
summarizer = TreeSummarize(verbose=True, streaming=True)

response = summarizer.get_response("Who is Jethro?", texts)

logger.newline()
logger.newline()

output = ""
for chunk in response:
    output += chunk
    logger.success(chunk, flush=True)


logger.info("\n\n[DONE]", bright=True)
