import os

from jet.logger.timer import time_it
from tqdm import tqdm
from jet.cache.joblib.utils import load_data, save_data
from jet.file.utils import save_file
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import logger
# from jet.llm.ollama import initialize_ollama_settings
import matplotlib.pyplot as plt
from jet.token.token_utils import get_ollama_tokenizer, token_counter
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.utils import set_global_tokenizer
# import tiktoken
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.llm.ollama.base_langchain import ChatOllama
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# initialize_ollama_settings()

embed_model = "snowflake-arctic-embed:137m"
llm_model = "llama3.1"

model_max_tokens = OLLAMA_MODEL_EMBEDDING_TOKENS[embed_model]
max_tokens = 0.5
chunk_size = int(model_max_tokens * max_tokens)
chunk_overlap = 40

embed_tokenizer = get_ollama_tokenizer(embed_model)
set_global_tokenizer(embed_tokenizer)

# pip install -U langchain umap-learn scikit-learn langchain_community tiktoken langchain-openai langchainhub langchain-chroma langchain-anthropic

"""
# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

The [RAPTOR](https://arxiv.org/pdf/2401.18059.pdf) paper presents an interesting approaching for indexing and retrieval of documents:

* The `leafs` are a set of starting documents
* Leafs are embedded and clustered
* Clusters are then summarized into higher level (more abstract) consolidations of information across similar documents

This process is done recursivly, resulting in a "tree" going from raw docs (`leafs`) to more abstract summaries.
 
We can applying this at varying scales; `leafs` can be:

* Text chunks from a single doc (as shown in the paper)
* Full docs (as we show below)

With longer context LLMs, it's possible to perform this over full documents. 

![Screenshot 2024-03-04 at 12.45.25 PM.png](attachment:72039e0c-e8c4-4b17-8780-04ad9fc584f3.png)

### Docs

Let's apply this to LangChain's LCEL documentation.

In this case, each `doc` is a unique web page of the LCEL docs.

The context varies from < 2k tokens on up to > 10k tokens.
"""

docs_path = "generated/RAPTOR/retrieved_docs.pkl"
docs: list[Document]

if os.path.exists(docs_path):
    logger.debug(f"Loading docs from cache: {docs_path}")
    docs = load_data(docs_path)
else:
    logger.info(f"Scraping docs from urls...")

    url = "https://en.wikipedia.org/wiki/I'll_Become_a_Villainess_Who_Goes_Down_in_History"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()

    url = "https://www.sportskeeda.com/anime/i-ll-become-villainess-who-goes-down-history-complete-release-schedule"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs_pydantic = loader.load()

    url = "https://www.animenewsnetwork.com/encyclopedia/anime.php?id=29354"
    loader = RecursiveUrlLoader(
        url=url, max_depth=3, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs_sq = loader.load()

    docs.extend([*docs_pydantic, *docs_sq])

    save_data(docs_path, docs)


docs_texts = [d.page_content for d in docs]
docs_texts_path = "generated/RAPTOR/docs_texts.json"
save_file(docs_texts, docs_texts_path)

# d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
# d_reversed = list(reversed(d_sorted))
# concatenated_content = "\n\n\n --- \n\n\n".join(
#     [doc.page_content for doc in d_reversed]
# )


splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    tokenizer=embed_tokenizer.encode,
)

# def length_function(text):
#     return token_counter(text, embed_model)
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap,
#     length_function=length_function,
# )
# texts_split = text_splitter.split_documents(docs)
docs_texts = splitter.split_texts(docs_texts)

splitted_docs_texts_path = "generated/RAPTOR/splitted_docs_texts.json"
save_file(docs_texts, splitted_docs_texts_path)

counts: list[int] = token_counter(docs_texts, embed_model, prevent_total=True)
# Compute min and max token counts
min_count = min(counts)
max_count = max(counts)

# Log token statistics
logger.debug(f"Min tokens in context: {min_count}")
logger.debug(f"Max tokens in context: {max_count}")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.title("Histogram of Token Counts")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)

# Add min/max annotations
plt.axvline(min_count, color='red', linestyle='dashed',
            linewidth=1, label=f"Min: {min_count}")
plt.axvline(max_count, color='green', linestyle='dashed',
            linewidth=1, label=f"Max: {max_count}")
plt.legend()

# Save the figure
histogram_path = "generated/RAPTOR/histogram.png"
plt.savefig(histogram_path, dpi=300, bbox_inches="tight")

# Close the figure to free memory
plt.close()

logger.log("Saved histograms to:", histogram_path,
           colors=["SUCCESS", "BRIGHT_SUCCESS"])


"""
## Models

We can test various models, including the new [Claude3](https://www.anthropic.com/news/claude-3-family) family.

Be sure to set the relevant API keys:

# * `ANTHROPIC_API_KEY`
# * `OPENAI_API_KEY`
"""


embd = OllamaEmbeddings(model=embed_model)


# Chat with summary context clusters

model = ChatOllama(model=llm_model)
llm_tokenizer = get_ollama_tokenizer(llm_model)
set_global_tokenizer(llm_tokenizer)

"""
### Tree Constrution

The clustering approach in tree construction includes a few interesting ideas.

**GMM (Gaussian Mixture Model)** 

- Model the distribution of data points across different clusters
- Optimal number of clusters by evaluating the model's Bayesian Information Criterion (BIC)

**UMAP (Uniform Manifold Approximation and Projection)** 

- Supports clustering
- Reduces the dimensionality of high-dimensional data
- UMAP helps to highlight the natural grouping of data points based on their similarities

**Local and Global Clustering** 

- Used to analyze data at different scales
- Both fine-grained and broader patterns within the data are captured effectively

**Thresholding** 

- Apply in the context of GMM to determine cluster membership
- Based on the probability distribution (assignment of data points to ≥ 1 cluster)
---

Code for GMM and thresholding is from Sarthi et al, as noted in the below two sources:
 
* [Origional repo](https://github.com/parthsarthi03/raptor/blob/master/raptor/cluster_tree_builder.py)
* [Minor tweaks](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raptor/llama_index/packs/raptor/clustering.py)

Full credit to both authors.
"""


RANDOM_SEED = 224  # Fixed seed for reproducibility


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
    """
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0])
                              for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def embed(texts):
    """
    Generate embeddings for a list of text documents.

    This function assumes the existence of an `embd` object with a method `embed_documents`
    that takes a list of texts and returns their embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    text_embeddings = embd.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


@time_it
def embed_cluster_texts(texts):
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    text_embeddings_np = embed(texts)  # Generate embeddings
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1
    )  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    # Store embeddings as a list in the DataFrame
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels  # Store cluster labels
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def embed_cluster_summarize_texts(
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """

    #  Cluster texts

    df_clusters = embed_cluster_texts(texts)

    df_clusters_dict = df_clusters.to_dict(orient="records")
    df_clusters_texts = [item["text"] for item in df_clusters_dict]

    tokens_llm = token_counter(
        df_clusters_texts, llm_model, prevent_total=True)
    tokens_embed = token_counter(
        df_clusters_texts, embed_model, prevent_total=True)

    df_clusters_list = [
        {
            "text": item["text"],
            "cluster": item["cluster"],
            "tokens": {
                "llm": tokens_llm[idx],
                "embed": tokens_embed[idx],
            },
            "embd": item["embd"],
        }
        for idx, item in enumerate(df_clusters_dict)
    ]

    clusters_path = "generated/RAPTOR/clusters.json"
    save_file(df_clusters_list, clusters_path)

    # Expand clusters

    expanded_list = []

    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    expanded_clusters_texts = [item["text"] for item in expanded_list]

    tokens_llm = token_counter(
        expanded_clusters_texts, llm_model, prevent_total=True)
    tokens_embed = token_counter(
        expanded_clusters_texts, embed_model, prevent_total=True)

    expanded_clusters_list = [
        {
            "text": item["text"],
            "cluster": item["cluster"],
            "tokens": {
                "llm": tokens_llm[idx],
                "embed": tokens_embed[idx],
            },
            "embd": item["embd"],
        }
        for idx, item in enumerate(expanded_list)
    ]

    expanded_clusters_path = "generated/RAPTOR/expanded_clusters.json"
    save_file(expanded_list, expanded_clusters_path)

    expanded_df = pd.DataFrame(expanded_list)
    all_clusters = expanded_df["cluster"].unique()

    all_clusters_path = "generated/RAPTOR/all_clusters.json"
    save_file(all_clusters, all_clusters_path)

    def run_summarize_clusters():
        logger.debug(f"--Generated {len(all_clusters)} clusters--")

        logger.log(
            "Save clusters to:",
            clusters_path,
            colors=["SUCCESS", "BRIGHT_SUCCESS"]
        )

        template = """Here is a sub-set of an unstructured text from a scraped page that may contain Anime data. 
        
        Give a detailed summary of the documentation provided.
        
        Documentation:
        {context}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model | StrOutputParser()

        @time_it
        def summarize_text(text: str):
            generated_summary = chain.invoke({"context": text})
            yield generated_summary

        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = fmt_txt(df_cluster)

            token_count: int = token_counter(formatted_txt, llm_model)
            model_max_length = OLLAMA_MODEL_EMBEDDING_TOKENS[llm_model]
            chunk_size = int(model_max_length * max_tokens)

            if token_count > chunk_size:
                warning = f"token_count ({token_count}) must be less than chunk size ({chunk_size}) for model ({model})"
                logger.warning(warning)
                splitter = SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    tokenizer=embed_tokenizer.encode,
                )
                splitted_texts = splitter.split_texts(formatted_txt)
            else:
                splitted_texts = [formatted_txt]

            for text in splitted_texts:
                token_count: int = token_counter(text, llm_model)
                logger.debug(f"Summarizing text ({token_count})...")

                # generated_summary = chain.invoke({"context": text})
                # yield generated_summary

                yield from summarize_text(text)

    summaries = []
    total_steps = len(all_clusters)
    progress_bar = tqdm(total=total_steps, desc="Summarizing clusters")
    for summary in run_summarize_clusters():
        summaries.append(summary)

        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters[:len(summaries)]),
            }
        )
        summary_clusters_path = "generated/RAPTOR/summary_clusters.json"
        logger.log(
            "Save summary clusters to:",
            summary_clusters_path,
            colors=["SUCCESS", "BRIGHT_SUCCESS"]
        )
        df_summary.to_json(summary_clusters_path, orient="records", indent=2)

        # Manually update progress after summarizing each cluster
        progress_bar.update(1)

    # Close progress bar after completion
    progress_bar.close()

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
    """
    results = {}  # Dictionary to store results at each level

    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

    results[level] = (df_clusters, df_summary)

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        results.update(next_level_results)

    return results


leaf_texts = docs_texts
logger.newline()
logger.info(f"Leaf Texts ({len(leaf_texts)}):")

leaf_texts_path = "generated/RAPTOR/leaf_texts.json"
save_file(leaf_texts, leaf_texts_path)

# Summarize clusters

logger.info("Summarizing leaf texts...")
results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)

logger.newline()
logger.debug(f"Result - recursive_embed_cluster_summarize ({results}):")

all_summary_clusters_path = "generated/RAPTOR/all_summary_clusters.json"
save_file(results, all_summary_clusters_path)

"""
The paper reports best performance from `collapsed tree retrieval`. 

This involves flattening the tree structure into a single layer and then applying a k-nearest neighbors (kNN) search across all nodes simultaneously. 

We do simply do this below.
"""


leaf_texts = leaf_texts.copy()

summary_texts = []

for level in sorted(results.keys()):
    summaries = results[level][1]["summaries"].tolist()
    summary_texts.extend(summaries)

logger.newline()
logger.info(f"Summary Texts ({len(summary_texts)}):")

summary_texts_path = "generated/RAPTOR/summary_texts.json"
save_file(summary_texts, summary_texts_path)

all_texts = [*leaf_texts, *summary_texts]

logger.newline()
logger.info(f"All Texts ({len(all_texts)}):")

all_texts_path = "generated/RAPTOR/all_texts.json"
save_file(all_texts, all_texts_path)


vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)
retriever = vectorstore.as_retriever()

"""
Now we can using our flattened, indexed tree in a RAG chain.
"""


prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

query = "How many seasons and episodes does ”I’ll Become a Villainess Who Goes Down in History” anime have?"
result = rag_chain.invoke(query)
logger.newline()
logger.debug("Result - rag_chain.invoke:")
logger.debug(query)
logger.success(result)

query_result_path = "generated/RAPTOR/query_result.json"
save_file({
    "query": query,
    "result": result
}, query_result_path)

"""
Trace: 

https://smith.langchain.com/public/1dabf475-1675-4494-b16c-928fbf079851/r
"""

logger.info("\n\n[DONE]", bright=True)
