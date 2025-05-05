import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.postgres import PostgresVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
import openai
import psycopg2
from typing import List, Dict
import nltk
nltk.download('punkt')

# Initialize OpenAI API (replace with your key)
openai.api_key = "your-openai-api-key"


def generate_metadata(text: str) -> Dict:
    """Generate metadata (keywords, summary) for a chunk using OpenAI."""
    prompt = f"Extract up to 6 keywords and a 1-sentence summary from the following text:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    result = response.choices[0].message.content
    keywords = result.split("\n")[0].replace("Keywords: ", "").split(", ")[:6]
    summary = result.split("\n")[1].replace("Summary: ", "")
    return {"keywords": keywords, "summary": summary}


def chunk_text(text: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Chunk text dynamically with overlap and attach metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
        length_function=len
    )
    chunks = splitter.split_text(text)
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = generate_metadata(chunk)
        metadata.update(
            {"source": source, "chunk_id": str(uuid.uuid4()), "position": i})
        doc = Document(page_content=chunk, metadata=metadata)
        documents.append(doc)
    return documents


def setup_vector_store(db_params: Dict) -> PostgresVectorStore:
    """Set up PostgresML vector store."""
    conn = psycopg2.connect(**db_params)
    vector_store = PostgresVectorStore.from_params(
        database=db_params["dbname"],
        host=db_params["host"],
        port=db_params["port"],
        user=db_params["user"],
        password=db_params["password"],
        table_name="rag_chunks"
    )
    return vector_store


def index_documents(documents: List[Document], vector_store: PostgresVectorStore):
    """Index documents into the vector store."""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model="text-embedding-ada-002"
    )
    return index


def retrieve_and_rerank(query: str, index: VectorStoreIndex, top_k: int = 100, rerank_top_k: int = 5) -> List[Dict]:
    """Retrieve documents and rerank using a cross-encoder."""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    reranker = LLMRerank(
        top_n=rerank_top_k,
        model="mixedbread-ai/mxbai-rerank-base-v1"
    )
    initial_nodes = retriever.retrieve(query)
    reranked_nodes = reranker.postprocess_nodes(initial_nodes, query=query)
    results = [
        {
            "text": node.node.text,
            "score": node.score,
            "metadata": node.node.metadata
        }
        for node in reranked_nodes
    ]
    return results


def main():
    # Database connection parameters
    db_params = {
        "dbname": "rag_db",
        "user": "postgres",
        "password": "your-password",
        "host": "localhost",
        "port": "5432"
    }

    # Real-world input examples
    financial_report = """
    Q3 2024 Financial Results: Company XYZ reported a revenue of $1.2 billion, up 15% year-over-year.
    Net income was $200 million, with a margin of 16.7%. The growth was driven by strong demand in the cloud computing segment, contributing $800 million to revenue.
    Operating expenses rose by 10% due to increased R&D investments.
    """
    tech_doc = """
    Microservices Architecture: A microservices architecture structures an application as a collection of loosely coupled services.
    Each service is independently deployable and communicates via APIs. Benefits include scalability and flexibility.
    Challenges include managing distributed systems and ensuring data consistency.
    """

    # Chunk documents
    financial_chunks = chunk_text(financial_report, "financial_report_2024")
    tech_chunks = chunk_text(tech_doc, "microservices_guide")
    all_chunks = financial_chunks + tech_chunks

    # Set up vector store and index
    vector_store = setup_vector_store(db_params)
    index = index_documents(all_chunks, vector_store)

    # Example queries
    queries = [
        "What drove the revenue growth for Company XYZ in Q3 2024?",
        "What are the benefits of microservices architecture?"
    ]

    # Retrieve and rerank for each query
    for query in queries:
        print(f"\nQuery: {query}")
        results = retrieve_and_rerank(query, index)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['text'][:100]}...")
            print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    main()
