from jet.llm.ollama.base_langchain import AzureChatOllama, AzureOllamaEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_cohere import CohereRerank
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import tqdm as tqdm
import uuid
import warnings

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Contextual Retrieval

In this notebook we will showcase how you can implement Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) using LangChain. Contextual Retrieval addresses the conundrum of traditional RAG approaches by prepending chunk-specific explanatory context to each chunk before embedding.

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F2496e7c6fedd7ffaa043895c23a4089638b0c21b-3840x2160.png&w=3840&q=75)
"""
logger.info("# Contextual Retrieval")


logging.disable(level=logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# os.environ["AZURE_OPENAI_API_KEY"] = "<YOUR_AZURE_OPENAI_API_KEY>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "<YOUR_AZURE_OPENAI_ENDPOINT>"
os.environ["COHERE_API_KEY"] = "<YOUR_COHERE_API_KEY>"

# !pip install -q langchain langchain-openai langchain-community faiss-cpu rank_bm25 langchain-cohere


"""
## Download Data

We will use `Paul Graham Essay` dataset.
"""
logger.info("## Download Data")

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O './paul_graham_essay.txt'

"""
#
#
 
S
e
t
u
p
 
L
L
M
 
a
n
d
 
E
m
b
e
d
d
i
n
g
 
m
o
d
e
l
"""
logger.info("#")

llm = AzureChatOllama(
    deployment_name="gpt-4-32k-0613",
    openai_api_version="2023-08-01-preview",
    temperature=0.0,
)

embeddings = AzureOllamaEmbeddings(
    deployment="text-embedding-ada-002",
    api_version="2023-08-01-preview",
)

"""
#
#
 
L
o
a
d
 
D
a
t
a
"""
logger.info("#")

loader = TextLoader("./paul_graham_essay.txt")
documents = loader.load()
WHOLE_DOCUMENT = documents[0].page_content

"""
## Prompts for creating context for each chunk

We will use the following prompts to create chunk-specific explanatory context to each chunk before embedding.
"""
logger.info("## Prompts for creating context for each chunk")

prompt_document = PromptTemplate(
    input_variables=["WHOLE_DOCUMENT"], template="{WHOLE_DOCUMENT}"
)
prompt_chunk = PromptTemplate(
    input_variables=["CHUNK_CONTENT"],
    template="Here is the chunk we want to situate within the whole document\n\n{CHUNK_CONTENT}\n\n"
    "Please give a short succinct context to situate this chunk within the overall document for "
    "the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.",
)

"""
#
#
 
R
e
t
r
i
e
v
e
r
s
"""
logger.info("#")



def split_text(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=200)
    doc_chunks = text_splitter.create_documents(texts)
    for i, doc in enumerate(doc_chunks):
        doc.metadata = {"doc_id": f"doc_{i}"}
    return doc_chunks


def create_embedding_retriever(documents_):
    vector_store = FAISS.from_documents(documents_, embedding=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4})


def create_bm25_retriever(documents_):
    retriever = BM25Retriever.from_documents(documents_, language="english")
    return retriever


class EmbeddingBM25RerankerRetriever:
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        reranker: BaseDocumentCompressor,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

    def invoke(self, query: str):
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        combined_docs = vector_docs + [
            doc for doc in bm25_docs if doc not in vector_docs
        ]

        reranked_docs = self.reranker.compress_documents(combined_docs, query)
        return reranked_docs

"""
#
#
#
 
N
o
n
-
c
o
n
t
e
x
t
u
a
l
 
r
e
t
r
i
e
v
e
r
s
"""
logger.info("#")

chunks = split_text([WHOLE_DOCUMENT])

embedding_retriever = create_embedding_retriever(chunks)

bm25_retriever = create_bm25_retriever(chunks)

reranker = CohereRerank(top_n=3, model="rerank-english-v2.0")

embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    vector_retriever=embedding_retriever,
    bm25_retriever=bm25_retriever,
    reranker=reranker,
)

"""
#
#
#
 
C
o
n
t
e
x
t
u
a
l
 
R
e
t
r
i
e
v
e
r
s
"""
logger.info("#")



def create_contextual_chunks(chunks_):
    contextual_documents = []
    for chunk in tqdm.tqdm(chunks_):
        context = prompt_document.format(WHOLE_DOCUMENT=WHOLE_DOCUMENT)
        chunk_context = prompt_chunk.format(CHUNK_CONTENT=chunk)
        llm_response = llm.invoke(context + chunk_context).content
        page_content = f"""Text: {chunk.page_content}\n\n\nContext: {llm_response}"""
        doc = Document(page_content=page_content, metadata=chunk.metadata)
        contextual_documents.append(doc)
    return contextual_documents


contextual_documents = create_contextual_chunks(chunks)

logger.debug(contextual_documents[1].page_content, "------------", chunks[1].page_content)

contextual_embedding_retriever = create_embedding_retriever(contextual_documents)

contextual_bm25_retriever = create_bm25_retriever(contextual_documents)

contextual_embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
    vector_retriever=contextual_embedding_retriever,
    bm25_retriever=contextual_bm25_retriever,
    reranker=reranker,
)

"""
#
#
 
G
e
n
e
r
a
t
e
 
Q
u
e
s
t
i
o
n
-
C
o
n
t
e
x
t
 
p
a
i
r
s
"""
logger.info("#")



DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and no prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""


class QuestionContextEvalDataset(BaseModel):
    """Embedding QA Dataset.
    Args:
        queries (Dict[str, str]): Dict id -> query.
        corpus (Dict[str, str]): Dict id -> string.
        relevant_docs (Dict[str, List[str]]): Dict query id -> list of doc ids.
    """

    queries: Dict[str, str]  # dict id -> query
    corpus: Dict[str, str]  # dict id -> string
    relevant_docs: Dict[str, List[str]]  # query id -> list of doc ids
    mode: str = "text"

    @property
    def query_docid_pairs(self) -> List[Tuple[str, List[str]]]:
        """Get query, relevant doc ids."""
        return [
            (query, self.relevant_docs[query_id])
            for query_id, query in self.queries.items()
        ]

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "QuestionContextEvalDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def generate_question_context_pairs(
    documents: List[Document],
    llm,
    qa_generate_prompt_tmpl: str = DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
) -> QuestionContextEvalDataset:
    """Generate evaluation dataset using watsonx LLM and a set of chunks with their chunk_ids

    Args:
        documents (List[Document]): chunks of data with chunk_id
        llm: LLM used for generating questions
        qa_generate_prompt_tmpl (str): prompt template used for generating questions
        num_questions_per_chunk (int): number of questions generated per chunk

    Returns:
        List[Documents]: List of langchain document objects with page content and metadata
    """
    doc_dict = {doc.metadata["doc_id"]: doc.page_content for doc in documents}
    queries = {}
    relevant_docs = {}
    for doc_id, text in tqdm(doc_dict.items()):
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.invoke(query).content
        result = re.split(r"\n+", response.strip())
        logger.debug(result)
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0][
            :num_questions_per_chunk
        ]

        num_questions_generated = len(questions)
        if num_questions_generated < num_questions_per_chunk:
            warnings.warn(
                f"Fewer questions generated ({num_questions_generated}) "
                f"than requested ({num_questions_per_chunk})."
            )
        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [doc_id]
    return QuestionContextEvalDataset(
        queries=queries, corpus=doc_dict, relevant_docs=relevant_docs
    )

qa_pairs = generate_question_context_pairs(chunks, llm, num_questions_per_chunk=2)

"""
#
#
 
E
v
a
l
u
a
t
e
"""
logger.info("#")

def compute_hit_rate(expected_ids, retrieved_ids):
    """
    Args:
    expected_ids List[str]: The ground truth doc_id
    retrieved_ids List[str]: The doc_id from retrieved chunks

    Returns:
        float: hit rate as a decimal
    """
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    is_hit = any(id in expected_ids for id in retrieved_ids)
    return 1.0 if is_hit else 0.0


def compute_mrr(expected_ids, retrieved_ids):
    """
    Args:
    expected_ids List[str]: The ground truth doc_id
    retrieved_ids List[str]: The doc_id from retrieved chunks

    Returns:
        float: MRR score as a decimal
    """
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    for i, id in enumerate(retrieved_ids):
        if id in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(expected_ids, retrieved_ids):
    """
    Args:
    expected_ids List[str]: The ground truth doc_id
    retrieved_ids List[str]: The doc_id from retrieved chunks

    Returns:
        float: nDCG score as a decimal
    """
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    dcg = 0.0
    idcg = 0.0
    for i, id in enumerate(retrieved_ids):
        if id in expected_ids:
            dcg += 1.0 / (i + 1)
        idcg += 1.0 / (i + 1)
    return dcg / idcg



def extract_queries(dataset):
    values = []
    for value in dataset.queries.values():
        values.append(value)
    return values


def extract_doc_ids(documents_):
    doc_ids = []
    for doc in documents_:
        doc_ids.append(f"{doc.metadata['doc_id']}")
    return doc_ids


def evaluate(retriever, dataset):
    mrr_result = []
    hit_rate_result = []
    ndcg_result = []

    for i in tqdm(range(len(dataset.queries))):
        context = retriever.invoke(extract_queries(dataset)[i])

        expected_ids = dataset.relevant_docs[list(dataset.queries.keys())[i]]
        retrieved_ids = extract_doc_ids(context)
        mrr = compute_mrr(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
        hit_rate = compute_hit_rate(
            expected_ids=expected_ids, retrieved_ids=retrieved_ids
        )
        ndgc = compute_ndcg(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
        mrr_result.append(mrr)
        hit_rate_result.append(hit_rate)
        ndcg_result.append(ndgc)

    array2D = np.array([mrr_result, hit_rate_result, ndcg_result])
    mean_results = np.mean(array2D, axis=1)
    results_df = pd.DataFrame(mean_results)
    results_df.index = ["MRR", "Hit Rate", "nDCG"]
    return results_df

embedding_bm25_rerank_results = evaluate(embedding_bm25_retriever_rerank, qa_pairs)

contextual_embedding_bm25_rerank_results = evaluate(
    contextual_embedding_bm25_retriever_rerank, qa_pairs
)

embedding_retriever_results = evaluate(embedding_retriever, qa_pairs)

contextual_embedding_retriever_results = evaluate(
    contextual_embedding_retriever, qa_pairs
)

bm25_results = evaluate(bm25_retriever, qa_pairs)

contextual_bm25_results = evaluate(contextual_bm25_retriever, qa_pairs)

def display_results(name, eval_results):
    """Display results from evaluate."""

    metrics = ["MRR", "Hit Rate", "nDCG"]

    columns = {
        "Retrievers": [name],
        **{metric: val for metric, val in zip(metrics, eval_results.values)},
    }

    metric_df = pd.DataFrame(columns)

    return metric_df


pd.concat(
    [
        display_results("Embedding Retriever", embedding_retriever_results),
        display_results("BM25 Retriever", bm25_results),
        display_results(
            "Embedding + BM25 Retriever + Reranker",
            embedding_bm25_rerank_results,
        ),
    ],
    ignore_index=True,
    axis=0,
)

pd.concat(
    [
        display_results(
            "Contextual Embedding Retriever", contextual_embedding_retriever_results
        ),
        display_results("Contextual BM25 Retriever", contextual_bm25_results),
        display_results(
            "Contextual Embedding + BM25 Retriever + Reranker",
            contextual_embedding_bm25_rerank_results,
        ),
    ],
    ignore_index=True,
    axis=0,
)

logger.info("\n\n[DONE]", bright=True)