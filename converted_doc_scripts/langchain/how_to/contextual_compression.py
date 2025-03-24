from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from jet.llm.ollama.base_langchain import ChatOllama
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import LLMListwiseRerank
from jet.llm.ollama.base_langchain import ChatOllama
from langchain.retrievers.document_compressors import EmbeddingsFilter
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter

initialize_ollama_settings()

"""
# How to do retrieval with contextual compression

One challenge with [retrieval](/docs/concepts/retrieval/) is that usually you don't know the specific queries your document storage system will face when you ingest data into the system. This means that the information most relevant to a query may be buried in a document with a lot of irrelevant text. Passing that full document through your application can lead to more expensive LLM calls and poorer responses.

Contextual compression is meant to fix this. The idea is simple: instead of immediately returning retrieved documents as-is, you can compress them using the context of the given query, so that only the relevant information is returned. “Compressing” here refers to both compressing the contents of an individual document and filtering out documents wholesale.

To use the Contextual Compression Retriever, you'll need:

- a base [retriever](/docs/concepts/retrievers/)
- a Document Compressor

The Contextual Compression Retriever passes queries to the base retriever, takes the initial documents and passes them through the Document Compressor. The Document Compressor takes a list of documents and shortens it by reducing the contents of documents or dropping documents altogether.

## Get started
"""


def pretty_print_docs(docs):
    logger.debug(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" +
                d.page_content for i, d in enumerate(docs)]
        )
    )


"""
## Using a vanilla vector store retriever
Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can see that given an example question our retriever returns one or two relevant docs and a few irrelevant docs. And even the relevant docs have a lot of irrelevant information in them.
"""

data_file = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/langchain/docs/docs/how_to/state_of_the_union.txt"
documents = TextLoader(data_file).load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, OllamaEmbeddings(
    model="nomic-embed-text")).as_retriever()

docs = retriever.invoke(
    "What did the president say about Ketanji Brown Jackson")
pretty_print_docs(docs)

"""
## Adding contextual compression with an `LLMChainExtractor`
Now let's wrap our base retriever with a `ContextualCompressionRetriever`. We'll add an `LLMChainExtractor`, which will iterate over the initially returned documents and extract from each only the content that is relevant to the query.
"""


llm = ChatOllama(model="llama3.1", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)

"""
## More built-in compressors: filters
### `LLMChainFilter`
The `LLMChainFilter` is slightly simpler but more robust compressor that uses an LLM chain to decide which of the initially retrieved documents to filter out and which ones to return, without manipulating the document contents.
"""


_filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)

"""
### `LLMListwiseRerank`

[LLMListwiseRerank](https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.document_compressors.listwise_rerank.LLMListwiseRerank.html) uses [zero-shot listwise document reranking](https://arxiv.org/pdf/2305.02156) and functions similarly to `LLMChainFilter` as a robust but more expensive option. It is recommended to use a more powerful LLM.

Note that `LLMListwiseRerank` requires a model with the [with_structured_output](/docs/integrations/chat/) method implemented.
"""


llm = ChatOllama(model="llama3.1")

_filter = LLMListwiseRerank.from_llm(llm, top_n=1)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)

"""
### `EmbeddingsFilter`

Making an extra LLM call over each retrieved document is expensive and slow. The `EmbeddingsFilter` provides a cheaper and faster option by embedding the documents and query and only returning those documents which have sufficiently similar embeddings to the query.
"""


embeddings = OllamaEmbeddings(model="nomic-embed-text")
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings, similarity_threshold=0.76)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)

"""
## Stringing compressors and document transformers together
Using the `DocumentCompressorPipeline` we can also easily combine multiple compressors in sequence. Along with compressors we can add `BaseDocumentTransformer`s to our pipeline, which don't perform any contextual compression but simply perform some transformation on a set of documents. For example `TextSplitter`s can be used as document transformers to split documents into smaller pieces, and the `EmbeddingsRedundantFilter` can be used to filter out redundant documents based on embedding similarity between documents.

Below we create a compressor pipeline by first splitting our docs into smaller chunks, then removing redundant documents, and then filtering based on relevance to the query.
"""


splitter = CharacterTextSplitter(
    chunk_size=300, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(
    embeddings=embeddings, similarity_threshold=0.76)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)


logger.info("\n\n[DONE]", bright=True)
