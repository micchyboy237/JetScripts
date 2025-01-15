"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/contextual_retrieval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Contextual Retrieval

In this notebook we will demonstrate how you can implement [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) using LlamaIndex abstractions.

We will use:

1. `Paul Graham Essay` dataset.
2. Anthropic LLM for context creation for each chunk.
3. Ollama LLM for Synthetic query generation and embedding model.
4. CohereAI Reranker.
"""

"""
## Installation
"""

# !pip install -U llama-index llama-index-llms-anthropic llama-index-postprocessor-cohere-rerank llama-index-retrievers-bm25 stemmer


from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.anthropic import Anthropic
import os
import nest_asyncio
nest_asyncio.apply()

"""
## Setup API Keys
"""


# os.environ["ANTHROPIC_API_KEY"] = "<YOUR ANTHROPIC API KEY>"

# os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"

os.environ["COHERE_API_KEY"] = "<YOUR COHEREAI API KEY>"

"""
## Setup LLM and Embedding model
"""


llm_anthropic = Anthropic(model="claude-3-5-sonnet-20240620")


Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

"""
## Download Data
"""

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O './paul_graham_essay.txt'

"""
## Load Data
"""


documents = SimpleDirectoryReader(
    input_files=["./paul_graham_essay.txt"],
).load_data()

WHOLE_DOCUMENT = documents[0].text

"""
## Prompts for creating context for each chunk

We will utilize anthropic prompt caching for creating context for each chunk. If you haven’t explored our integration yet, please take a moment to review it [here](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/llm/anthropic_prompt_caching.ipynb).
"""

prompt_document = """<document>
{WHOLE_DOCUMENT}
</document>"""

prompt_chunk = """Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

"""
## Utils

1. `create_contextual_nodes` - Function to create contextual nodes for a list of nodes.

2. `create_embedding_retriever` - Function to create an embedding retriever for a list of nodes.

3. `create_bm25_retriever` - Function to create a bm25 retriever for a list of nodes.

4. `EmbeddingBM25RerankerRetriever` - Custom retriever that uses both embedding and bm25 retrievers and reranker.

5. `create_eval_dataset` - Function to create a evaluation dataset from a list of nodes.

6. `set_node_ids` - Function to set node ids for a list of nodes.

7. `retrieval_results` - Function to get retrieval results for a retriever and evaluation dataset.

8. `display_results` - Function to display results from `retrieval_results`
"""


def create_contextual_nodes(nodes_):
    """Function to create contextual nodes for a list of nodes"""
    nodes_modified = []
    for node in nodes_:
        new_node = copy.deepcopy(node)
        messages = [
            ChatMessage(role="system", content="You are helpful AI Assitant."),
            ChatMessage(
                role="user",
                content=[
                    TextBlock(
                        text=prompt_document.format(
                            WHOLE_DOCUMENT=WHOLE_DOCUMENT
                        )
                    ),
                    TextBlock(
                        text=prompt_chunk.format(CHUNK_CONTENT=node.text)
                    ),
                ],
                additional_kwargs={"cache_control": {"type": "ephemeral"}},
            ),
        ]
        new_node.metadata["context"] = str(
            llm_anthropic.chat(
                messages,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
        )
        nodes_modified.append(new_node)

    return nodes_modified


def create_embedding_retriever(nodes_, similarity_top_k=2):
    """Function to create an embedding retriever for a list of nodes"""
    vector_index = VectorStoreIndex(nodes_)
    retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
    return retriever


def create_bm25_retriever(nodes_, similarity_top_k=2):
    """Function to create a bm25 retriever for a list of nodes"""
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes_,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    return bm25_retriever


def create_eval_dataset(nodes_, llm, num_questions_per_chunk=2):
    """Function to create a evaluation dataset for a list of nodes"""
    qa_dataset = generate_question_context_pairs(
        nodes_, llm=llm, num_questions_per_chunk=num_questions_per_chunk
    )
    return qa_dataset


def set_node_ids(nodes_):
    """Function to set node ids for a list of nodes"""

    for index, node in enumerate(nodes_):
        node.id_ = f"node_{index}"

    return nodes_


async def retrieval_results(retriever, eval_dataset):
    """Function to get retrieval results for a retriever and evaluation dataset"""

    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metrics, retriever=retriever
    )

    eval_results = retriever_evaluator.evaluate_dataset(qa_dataset)

    return eval_results


def display_results(name, eval_results):
    """Display results from evaluate."""

    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    columns = {
        "retrievers": [name],
        **{k: [full_df[k].mean()] for k in metrics},
    }

    metric_df = pd.DataFrame(columns)

    return metric_df


class EmbeddingBM25RerankerRetriever(BaseRetriever):
    """Custom retriever that uses both embedding and bm25 retrievers and reranker"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        reranker: CohereRerank,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        vector_nodes.extend(bm25_nodes)

        retrieved_nodes = self.reranker.postprocess_nodes(
            vector_nodes, query_bundle
        )

        return retrieved_nodes


"""
## Create Nodes
"""


node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

"""
## Set node ids

Useful to have consistent result comparison for nodes with and without contextual text.
"""

nodes = set_node_ids(nodes)

nodes[0].metadata

"""
## Create contextual nodes
"""

nodes_contextual = create_contextual_nodes(nodes)

nodes[0].metadata, nodes_contextual[0].metadata

"""
## Set `similarity_top_k`
"""

similarity_top_k = 3

"""
## Set `CohereReranker`
"""


cohere_rerank = CohereRerank(
    api_key=os.environ["COHERE_API_KEY"], top_n=similarity_top_k
)

"""
## Create retrievers.

1. Embedding based retriever.
2. BM25 based retriever.
3. Embedding + BM25 + Cohere reranker retriever.
"""

embedding_retriever = create_embedding_retriever(
    nodes, similarity_top_k=similarity_top_k
)
bm25_retriever = create_bm25_retriever(
    nodes, similarity_top_k=similarity_top_k
)
embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    embedding_retriever, bm25_retriever, reranker=cohere_rerank
)

"""
## Create retrievers using contextual nodes.
"""

contextual_embedding_retriever = create_embedding_retriever(
    nodes_contextual, similarity_top_k=similarity_top_k
)
contextual_bm25_retriever = create_bm25_retriever(
    nodes_contextual, similarity_top_k=similarity_top_k
)
contextual_embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    contextual_embedding_retriever,
    contextual_bm25_retriever,
    reranker=cohere_rerank,
)

"""
## Create Synthetic query dataset
"""


llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)

qa_dataset = create_eval_dataset(nodes, llm=llm, num_questions_per_chunk=2)

list(qa_dataset.queries.values())[1]

"""
## Evaluate retrievers with and without contextual nodes
"""

embedding_retriever_results = await retrieval_results(
    embedding_retriever, qa_dataset
)
bm25_retriever_results = await retrieval_results(bm25_retriever, qa_dataset)
embedding_bm25_retriever_rerank_results = await retrieval_results(
    embedding_bm25_retriever_rerank, qa_dataset
)

contextual_embedding_retriever_results = await retrieval_results(
    contextual_embedding_retriever, qa_dataset
)
contextual_bm25_retriever_results = await retrieval_results(
    contextual_bm25_retriever, qa_dataset
)
contextual_embedding_bm25_retriever_rerank_results = await retrieval_results(
    contextual_embedding_bm25_retriever_rerank, qa_dataset
)

"""
## Display results
"""

"""
### Without Context
"""

pd.concat(
    [
        display_results("Embedding Retriever", embedding_retriever_results),
        display_results("BM25 Retriever", bm25_retriever_results),
        display_results(
            "Embedding + BM25 Retriever + Reranker",
            embedding_bm25_retriever_rerank_results,
        ),
    ],
    ignore_index=True,
    axis=0,
)

"""
### With Context
"""

pd.concat(
    [
        display_results(
            "Contextual Embedding Retriever",
            contextual_embedding_retriever_results,
        ),
        display_results(
            "Contextual BM25 Retriever", contextual_bm25_retriever_results
        ),
        display_results(
            "Contextual Embedding + Contextual BM25 Retriever + Reranker",
            contextual_embedding_bm25_retriever_rerank_results,
        ),
    ],
    ignore_index=True,
    axis=0,
)

"""
## Observation:

We observed improved metrics with contextual retrieval; however, our experiments showed that much depends on the queries, chunk size, chunk overlap, and other variables. Therefore, it’s essential to experiment to optimize the benefits of this technique.
"""
