from jet.models.config import MODELS_CACHE_DIR
from eval_utils import evaluate, display_results
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import LinearAdapterEmbeddingModel
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.embeddings.adapter_utils import BaseAdapter
from llama_index.core.embeddings.adapter_utils import TwoLayerNN
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.schema import TextNode
from llama_index.embeddings.adapter import AdapterEmbeddingModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.finetuning import generate_qa_embedding_pairs
from torch import nn, Tensor
from tqdm.notebook import tqdm
from typing import Dict
import json
import os
import pandas as pd
import shutil
import torch
import torch.nn.functional as F


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/finetuning/embeddings/finetune_embedding_adapter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Finetuning an Adapter on Top of any Black-Box Embedding Model


We have capabilities in LlamaIndex allowing you to fine-tune an adapter on top of embeddings produced from any model (sentence_transformers, OllamaFunctionCallingAdapter, and more). 

This allows you to transform your embedding representations into a new latent space that's optimized for retrieval over your specific data and queries. This can lead to small increases in retrieval performance that in turn translate to better performing RAG systems.

We do this via our `EmbeddingAdapterFinetuneEngine` abstraction. We fine-tune three types of adapters:
- Linear
- 2-Layer NN
- Custom NN

## Generate Corpus

We use our helper abstractions, `generate_qa_embedding_pairs`, to generate our training and evaluation dataset. This function takes in any set of text nodes (chunks) and generates a structured dataset containing (question, context) pairs.
"""
logger.info("# Finetuning an Adapter on Top of any Black-Box Embedding Model")

# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-embeddings-adapter
# %pip install llama-index-finetuning


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

TRAIN_FILES = [
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10k/lyft_2021.pdf"]
VAL_FILES = [
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10k/uber_2021.pdf"]

TRAIN_CORPUS_FPATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/train_corpus.json"
VAL_CORPUS_FPATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/val_corpus.json"


def load_corpus(files, verbose=False):
    if verbose:
        logger.debug(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        logger.debug(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        logger.debug(f"Parsed {len(nodes)} nodes")

    return nodes


"""
We do a very naive train/val split by having the Lyft corpus as the train dataset, and the Uber corpus as the val dataset.
"""
logger.info("We do a very naive train/val split by having the Lyft corpus as the train dataset, and the Uber corpus as the val dataset.")

train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

"""
### Generate synthetic queries

Now, we use an LLM (gpt-3.5-turbo) to generate questions using each text chunk in the corpus as context.

Each pair of (generated question, text chunk used as context) becomes a datapoint in the finetuning dataset (either for training or evaluation).
"""
logger.info("### Generate synthetic queries")


train_dataset = generate_qa_embedding_pairs(train_nodes)
val_dataset = generate_qa_embedding_pairs(val_nodes)

train_dataset.save_json("train_dataset.json")
val_dataset.save_json("val_dataset.json")

train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

"""
## Run Embedding Finetuning

We then fine-tune our linear adapter on top of an existing embedding model. We import our new `EmbeddingAdapterFinetuneEngine` abstraction, which takes in an existing embedding model and a set of training parameters.

#### Fine-tune bge-small-en (default)
"""
logger.info("## Run Embedding Finetuning")


base_embed_model = resolve_embed_model("local:BAAI/bge-small-en")

finetune_engine = EmbeddingAdapterFinetuneEngine(
    train_dataset,
    base_embed_model,
    model_output_path="model_output_test",
    epochs=4,
    verbose=True,
)

finetune_engine.finetune()

embed_model = finetune_engine.get_finetuned_model()


"""
## Evaluate Finetuned Model

We compare the fine-tuned model against the base model, as well as against text-embedding-ada-002.

We evaluate with two ranking metrics:
- **Hit-rate metric**: For each (query, context) pair, we retrieve the top-k documents with the query. It's a hit if the results contain the ground-truth context.
- **Mean Reciprocal Rank**: A slightly more granular ranking metric that looks at the "reciprocal rank" of the ground-truth context in the top-k retrieved set. The reciprocal rank is defined as 1/rank. Of course, if the results don't contain the context, then the reciprocal rank is 0.
"""
logger.info("## Evaluate Finetuned Model")


ada = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
ada_val_results = evaluate(val_dataset, ada)

display_results(["ada"], [ada_val_results])

bge = "local:BAAI/bge-small-en"
bge_val_results = evaluate(val_dataset, bge)

display_results(["bge"], [bge_val_results])

ft_val_results = evaluate(val_dataset, embed_model)

display_results(["ft"], [ft_val_results])

"""
Here we show all the results concatenated together.
"""
logger.info("Here we show all the results concatenated together.")

display_results(
    ["ada", "bge", "ft"], [ada_val_results, bge_val_results, ft_val_results]
)

"""
## Fine-tune a Two-Layer Adapter

Let's try fine-tuning a two-layer NN as well! 

It's a simple two-layer NN with a ReLU activation and a residual layer at the end.

We train for 25 epochs - longer than the linear adapter - and preserve checkpoints every 100 steps.
"""
logger.info("## Fine-tune a Two-Layer Adapter")


base_embed_model = resolve_embed_model("local:BAAI/bge-small-en")
adapter_model = TwoLayerNN(
    384,  # input dimension
    1024,  # hidden dimension
    384,  # output dimension
    bias=True,
    add_residual=True,
)

finetune_engine = EmbeddingAdapterFinetuneEngine(
    train_dataset,
    base_embed_model,
    model_output_path="model5_output_test",
    model_checkpoint_path="model5_ck",
    adapter_model=adapter_model,
    epochs=25,
    verbose=True,
)

finetune_engine.finetune()

embed_model_2layer = finetune_engine.get_finetuned_model(
    adapter_cls=TwoLayerNN
)

"""
### Evaluation Results

Run the same evaluation script used in the previous section to measure hit-rate/MRR within the two-layer model.
"""
logger.info("### Evaluation Results")

embed_model_2layer = AdapterEmbeddingModel(
    base_embed_model,
    "model5_output_test",
    TwoLayerNN,
)


ft_val_results_2layer = evaluate(val_dataset, embed_model_2layer)

display_results(
    ["ada", "bge", "ft_2layer"],
    [ada_val_results, bge_val_results, ft_val_results_2layer],
)

embed_model_2layer_s900 = AdapterEmbeddingModel(
    base_embed_model,
    "model5_ck/step_900",
    TwoLayerNN,
)

ft_val_results_2layer_s900 = evaluate(val_dataset, embed_model_2layer_s900)

display_results(
    ["ada", "bge", "ft_2layer_s900"],
    [ada_val_results, bge_val_results, ft_val_results_2layer_s900],
)

"""
## Try Your Own Custom Model

You can define your own custom adapter here! Simply subclass `BaseAdapter`, which is a light wrapper around the `nn.Module` class.

You just need to subclass `forward` and `get_config_dict`.

Just make sure you're familiar with writing `PyTorch` code :)
"""
logger.info("## Try Your Own Custom Model")


class CustomNN(BaseAdapter):
    """Custom NN transformation.

    Is a copy of our TwoLayerNN, showing it here for notebook purposes.

    Args:
        in_features (int): Input dimension.
        hidden_features (int): Hidden dimension.
        out_features (int): Output dimension.
        bias (bool): Whether to use bias. Defaults to False.
        activation_fn_str (str): Name of activation function. Defaults to "relu".

    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = False,
        add_residual: bool = False,
    ) -> None:
        super(CustomNN, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias

        self.linear1 = nn.Linear(in_features, hidden_features, bias=True)
        self.linear2 = nn.Linear(hidden_features, out_features, bias=True)
        self._add_residual = add_residual
        self.residual_weight = nn.Parameter(torch.zeros(1))

    def forward(self, embed: Tensor) -> Tensor:
        """Forward pass (Wv).

        Args:
            embed (Tensor): Input tensor.

        """
        output1 = self.linear1(embed)
        output1 = F.relu(output1)
        output2 = self.linear2(output1)

        if self._add_residual:
            output2 = self.residual_weight * output2 + embed

        return output2

    def get_config_dict(self) -> Dict:
        """Get config dict."""
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "add_residual": self._add_residual,
        }


custom_adapter = CustomNN(
    384,  # input dimension
    1024,  # hidden dimension
    384,  # output dimension
    bias=True,
    add_residual=True,
)

finetune_engine = EmbeddingAdapterFinetuneEngine(
    train_dataset,
    base_embed_model,
    model_output_path="custom_model_output",
    model_checkpoint_path="custom_model_ck",
    adapter_model=custom_adapter,
    epochs=25,
    verbose=True,
)

finetune_engine.finetune()

embed_model_custom = finetune_engine.get_finetuned_model(
    adapter_cls=CustomAdapter
)

"""
### Evaluation Results

Run the same evaluation script used in the previous section to measure hit-rate/MRR.
"""
logger.info("### Evaluation Results")


ft_val_results_custom = evaluate(val_dataset, embed_model_custom)

display_results(["ft_custom"]x, [ft_val_results_custom])

logger.info("\n\n[DONE]", bright=True)
