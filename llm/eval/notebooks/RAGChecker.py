%pip install -qU ragchecker llama-indexfrom llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from ragchecker.integrations.llama_index import response_to_rag_results
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
documents = SimpleDirectoryReader("path/to/your/documents").load_data()

index = VectorStoreIndex.from_documents(documents)

rag_application = index.as_query_engine()
user_query = "What is RAGChecker?"
gt_answer = "RAGChecker is an advanced automatic evaluation framework designed to assess and diagnose Retrieval-Augmented Generation (RAG) systems. It provides a comprehensive suite of metrics and tools for in-depth analysis of RAG performance."


response_object = rag_application.query(user_query)

rag_result = response_to_rag_results(
    query=user_query,
    gt_answer=gt_answer,
    response_object=response_object,
)

rag_results = RAGResults.from_dict({"results": [rag_result]})
print(rag_results)
evaluator = RAGChecker(
    extractor_name="bedrock/meta.llama3-70b-instruct-v1:0",
    checker_name="bedrock/meta.llama3-70b-instruct-v1:0",
    batch_size_extractor=32,
    batch_size_checker=32,
)

evaluator.evaluate(rag_results, all_metrics)

print(rag_results)
from ragchecker.metrics import (
    overall_metrics,
    retriever_metrics,
    generator_metrics,
)
from ragchecker.metrics import (
    precision,
    recall,
    f1,
    claim_recall,
    context_precision,
    context_utilization,
    noise_sensitivity_in_relevant,
    noise_sensitivity_in_irrelevant,
    hallucination,
    self_knowledge,
    faithfulness,
)
