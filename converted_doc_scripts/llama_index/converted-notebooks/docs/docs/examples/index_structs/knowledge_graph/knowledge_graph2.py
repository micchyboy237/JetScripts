from IPython.display import HTML
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import download_loader
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.readers.papers import ArxivReader
from llama_index.readers.web import SimpleWebPageReader
from pyvis.network import Network
from transformers import pipeline
import logging
import openai
import os
import shutil
import sys
import wikipedia


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Knowledge Graph Construction w/ WikiData Filtering

In this notebook, we compare using [REBEL](https://huggingface.co/Babelscape/rebel-large) for knowledge graph construction with and without filtering from wikidata.

This is a simplified version, find out more about using wikipedia for filtering, check here
- [Make Meaningful Knowledge Graph from OpenSource REBEL Model](https://medium.com/@haiyangli_38602/make-meaningful-knowledge-graph-from-opensource-rebel-model-6f9729a55527)

## Setup
"""
logger.info("# Knowledge Graph Construction w/ WikiData Filtering")

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-web
# %pip install llama-index-readers-papers

# !pip install llama_index transformers wikipedia html2text pyvis


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
## 1. extract via huggingface pipeline

The initial pipeline uses the provided extraction code from the [HuggingFace model card](https://huggingface.co/Babelscape/rebel-large).
"""
logger.info("## 1. extract via huggingface pipeline")


triplet_extractor = pipeline(
    "text2text-generation",
    model="Babelscape/rebel-large",
    tokenizer="Babelscape/rebel-large",
    device="cuda:0",
)


def extract_triplets(input_text):
    text = triplet_extractor.tokenizer.batch_decode(
        [
            triplet_extractor(
                input_text, return_tensors=True, return_text=False
            )[0]["generated_token_ids"]
        ]
    )[0]

    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "")
        .replace("<pad>", "")
        .replace("</s>", "")
        .split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    (subject.strip(), relation.strip(), object_.strip())
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    (subject.strip(), relation.strip(), object_.strip())
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token

    if subject != "" and relation != "" and object_ != "":
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets


"""
## 2. Extract with wiki filtering

Optionally, we can filter our extracted relations using data from wikipedia.
"""
logger.info("## 2. Extract with wiki filtering")


class WikiFilter:
    def __init__(self):
        self.cache = {}

    def filter(self, candidate_entity):
        if candidate_entity in self.cache:
            return self.cache[candidate_entity]["title"]

        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary,
            }

            self.cache[candidate_entity] = entity_data
            self.cache[page.title] = entity_data

            return entity_data["title"]
        except:
            return None


wiki_filter = WikiFilter()


def extract_triplets_wiki(text):
    relations = extract_triplets(text)

    filtered_relations = []
    for relation in relations:
        (subj, rel, obj) = relation
        filtered_subj = wiki_filter.filter(subj)
        filtered_obj = wiki_filter.filter(obj)

        if filtered_subj is None and filtered_obj is None:
            continue

        filtered_relations.append(
            (
                filtered_subj or subj,
                rel,
                filtered_obj or obj,
            )
        )

    return filtered_relations


"""
## Run with Llama_Index
"""
logger.info("## Run with Llama_Index")


loader = ArxivReader()
documents = loader.load_data(
    search_query="Retrieval Augmented Generation", max_results=1
)


# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]


documents = [Document(text="".join([x.text for x in documents]))]


llm = OllamaFunctionCalling(temperature=0.1, model="llama3.2")
Settings.llm = llm
Settings.chunk_size = 256

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

"""
NOTE: This next cell takes about 4mins on GPU.
"""
logger.info("NOTE: This next cell takes about 4mins on GPU.")

index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=3,
    kg_triplet_extract_fn=extract_triplets,
    storage_context=storage_context,
    include_embeddings=True,
)

index1 = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=3,
    kg_triplet_extract_fn=extract_triplets_wiki,
    storage_context=storage_context,
    include_embeddings=True,
)


g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.save_graph("non_filtered_graph.html")


HTML(filename="non_filtered_graph.html")


g = index1.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.save_graph("wiki_filtered_graph.html")


HTML(filename="wiki_filtered_graph.html")

logger.info("\n\n[DONE]", bright=True)
