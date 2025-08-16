from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
Below is a comprehensive summary table consolidating the features and use cases for all 22 RAG-related documents discussed across the conversations. The table captures the key features and primary use cases for each document, organized for clarity and comparison.

---

**Comprehensive Summary Table of RAG Techniques**

| **Document**                                                     | **Key Features**                                                                                                                     | **Primary Use Cases**                                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| **Hierarchical Indices (18_hierarchy_rag.md)**                   | Two-tier retrieval with summaries, preserves context, evaluates vs. standard RAG                                                     | Large document collections, context-sensitive queries, efficient retrieval        |
| **Learning RAG Configurations (best_rag_finder.md)**             | Tests chunking parameters, top_k, and strategies (Simple, Query Rewrite, Rerank); evaluates with Faithfulness, Relevancy, Similarity | RAG optimization, educational tool, performance benchmarking                      |
| **Proposition Chunking (14_proposition_chunking.md)**            | Breaks text into atomic factual statements, ensures semantic integrity, filters low-quality propositions                             | Fact-based retrieval, granular knowledge bases, noise reduction                   |
| **Hypothetical Document Embedding (HyDE) (19_HyDE_rag.md)**      | Transforms queries into hypothetical documents, bridges semantic gap, compares with standard RAG                                     | Complex queries, semantic search, cross-domain retrieval                          |
| **Contextual Compression (10_contextual_compression.md)**        | Filters/compresses retrieved chunks for relevance, batch processing, compares with standard RAG                                      | Noise reduction, efficient context use, high-relevance outputs                    |
| **Query Transformations (7_query_transform.md)**                 | Query rewriting, step-back prompting, sub-query decomposition for enhanced retrieval                                                 | Complex query handling, contextual retrieval, precision improvement               |
| **Fusion Retrieval (16_fusion_rag.md)**                          | Combines vector and BM25 search, normalizes scores, weighted ranking                                                                 | Hybrid search needs, diverse query types, robust retrieval                        |
| **Feedback Loop RAG (11_feedback_loop_rag.md)**                  | Dynamic system with user feedback, adjusts relevance scores, integrates Q&A pairs                                                    | Continuous improvement, personalized knowledge bases, adaptive systems            |
| **Corrective RAG (CRAG) (20_crag.md)**                           | Evaluates and corrects retrieval, uses web search fallback, combines multiple sources                                                | Incomplete knowledge bases, high-reliability systems, dynamic information needs   |
| **Simple RAG (1_simple_rag.md)**                                 | Basic RAG with data_ingestion, chunking, semantic search, and evaluation                                                             | Beginner implementation, small-scale knowledge bases, baseline for comparison     |
| **Simple RAG with RL (21_rag_with_rl.md)**                       | RL-enhanced RAG with policy network, reward-based optimization (cosine similarity), compares with simple RAG                         | Accuracy optimization, adaptive retrieval, advanced RAG research                  |
| **Graph RAG (17_graph_rag.md)**                                  | Graph-based knowledge organization, concept traversal, visualization                                                                 | Complex query resolution, explainable AI, context-rich retrieval                  |
| **Context-Enriched RAG (4_context_enriched_rag.md)**             | Context-aware retrieval with neighboring chunks, ensures coherence                                                                   | Coherent answer generation, context-dependent queries, improved completeness      |
| **Reranking (8_reranker.md)**                                    | Refines retrieval with LLM/keyword-based reranking, improves relevance                                                               | Improved retrieval precision, noise reduction, enhanced response quality          |
| **Adaptive Retrieval (12_adaptive_rag.md)**                      | Query-type classification (Factual, Analytical, Opinion, Contextual), specialized retrieval strategies                               | Diverse query handling, personalized retrieval, high accuracy across domains      |
| **Chunk Size Selector (3_chunk_size_selector.md)**               | Evaluates chunk size impact, compares faithfulness/relevancy                                                                         | RAG optimization, performance benchmarking, educational tool                      |
| **Self-RAG (13_self_rag.md)**                                    | Dynamic retrieval decisions, relevance/support/utility evaluations                                                                   | Efficient retrieval, reliable responses, dynamic query handling                   |
| **Relevant Segment Extraction (RSE) (9_rse.md)**                 | Reconstructs continuous segments from clustered chunks, preserves context                                                            | Coherent context retrieval, comprehensive answers, document analysis              |
| **Multi-Modal RAG (15_multimodel_rag.md)**                       | Incorporates text and image data, generates image captions, compares with text-only RAG                                              | Visual data integration, comprehensive knowledge bases, enhanced query answering  |
| **Semantic Chunking (2_semantic_chunking.md)**                   | Splits text based on content similarity (percentile method), evaluates performance                                                   | Meaningful text segmentation, improved retrieval accuracy, research/prototyping   |
| **Document Augmentation RAG (6_doc_augmentation_rag.md)**        | Generates questions for chunks, embeds questions and text, enhances retrieval                                                        | Enhanced retrieval, knowledge base enrichment, query flexibility                  |
| **Contextual Chunk Headers (5_contextual_chunk_headers_rag.md)** | Prepends headers to chunks, improves context in retrieval                                                                            | Context-aware retrieval, accurate response generation, structured data processing |

---

**Notes:**

- Each document focuses on a unique enhancement to the RAG framework, addressing specific challenges such as retrieval accuracy, context preservation, query complexity, or multi-modal data integration.
- The use cases reflect the practical applications of each technique, ranging from educational tools and prototyping to advanced systems for enterprise, research, or real-time applications.
- Common components across documents include PDF text extraction (often using PyMuPDF), embedding creation (typically with Ollama API), simple vector stores, and response generation/evaluation, tailored to the specific technique.

This table provides a holistic view of the RAG techniques, enabling users to select the most appropriate method based on their dataset, query types, and performance goals.
"""
logger.info("Below is a comprehensive summary table consolidating the features and use cases for all 22 RAG-related documents discussed across the conversations. The table captures the key features and primary use cases for each document, organized for clarity and comparison.")

logger.info("\n\n[DONE]", bright=True)