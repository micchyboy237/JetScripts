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
For searching unstructured web-scraped content with queries resembling typical Google searches (often short, keyword-heavy, or intent-driven), certain RAG techniques from the provided documents are better suited due to their ability to handle noisy, diverse, and unstructured data while accommodating the varied nature of Google-like queries. Below, I outline the most suitable techniques, their relevant features, and why they are effective for this scenario, followed by a brief explanation of why others may be less optimal.

---

### Recommended RAG Techniques for Unstructured Web-Scraped Content with Google-Like Queries

1. **Fusion Retrieval (16_fusion_rag.md)**  
   **Key Features**: Combines semantic vector search (for conceptual understanding) with keyword-based BM25 retrieval (for exact term matching), normalizes scores, and ranks documents using a weighted formula.  
   **Why It Works**:

   - Web-scraped content is often noisy and contains a mix of relevant and irrelevant text. Fusion retrieval leverages both semantic similarity (to capture intent) and keyword matching (to align with specific terms in Google-like queries).
   - Google searches often include exact keywords or phrases (e.g., "best budget laptop 2025"), which BM25 excels at matching, while vector search handles broader intent (e.g., "budget-friendly laptops").
   - The weighted ranking ensures robust retrieval across diverse query types, making it ideal for unstructured, heterogeneous web data.  
     **Use Case Fit**: Perfect for hybrid search needs, such as web content aggregation or news archive searches, where queries range from specific to conceptual.

2. **Query Transformations (7_query_transform.md)**  
   **Key Features**: Implements query rewriting (for specificity), step-back prompting (for broader context), and sub-query decomposition (for complex queries).  
   **Why It Works**:

   - Google-like queries are often vague, ambiguous, or complex (e.g., "why is my laptop slow"). Query transformations refine these into more precise or comprehensive forms, improving retrieval from unstructured web content.
   - Sub-query decomposition breaks down multi-part queries (e.g., "laptop slow and overheating") into simpler components, ensuring comprehensive coverage of relevant web-scraped data.
   - Step-back prompting retrieves broader context, which is useful for poorly structured web data lacking clear organization.  
     **Use Case Fit**: Ideal for handling complex or ambiguous Google-like queries in customer support or general knowledge search systems.

3. **Contextual Compression (10_contextual_compression.md)**  
   **Key Features**: Filters and compresses retrieved chunks to retain only query-relevant information, reducing noise and optimizing the context window.  
   **Why It Works**:

   - Web-scraped content often includes irrelevant sections (e.g., ads, navigation menus). Contextual compression ensures that only the most relevant text is used, improving response quality.
   - Google-like queries benefit from concise, focused context, as users expect quick, relevant answers rather than verbose outputs.
   - Batch compression enhances efficiency, critical for processing large volumes of unstructured web data.  
     **Use Case Fit**: Effective for noise reduction in web-based Q&A systems or chatbots dealing with scraped content.

4. **Reranking (8_reranker.md)**  
   **Key Features**: Refines initial retrieval with LLM-based or keyword-based reranking, scoring and reordering documents for relevance.  
   **Why It Works**:

   - Initial retrieval from unstructured web content may include semi-relevant chunks due to noise. Reranking ensures the most relevant documents are prioritized, aligning with the precision expected from Google-like searches.
   - Keyword-based reranking complements Google-style queries, which often rely on specific terms, while LLM-based reranking enhances semantic relevance.
   - It acts as a second filter, improving the quality of retrieved content from messy web data.  
     **Use Case Fit**: Suitable for applications requiring high retrieval precision, such as web content summarization or targeted information extraction.

5. **Corrective RAG (CRAG) (20_crag.md)**  
   **Key Features**: Evaluates retrieved content for relevance, corrects retrieval with web search fallback, and combines multiple sources dynamically.  
   **Why It Works**:
   - Unstructured web-scraped content may lack complete or relevant information for certain queries. CRAG’s web search fallback ensures missing information is supplemented, aligning with Google’s ability to pull from diverse sources.
   - Relevance evaluation filters out low-quality or irrelevant scraped content, critical for noisy datasets.
   - Dynamic source combination mimics Google’s multi-source approach, making it effective for varied query intents.  
     **Use Case Fit**: Ideal for scenarios with incomplete scraped data, such as real-time news aggregators or research assistants.

---

### Why These Techniques Are Best Suited

- **Handling Unstructured Data**: Web-scraped content is typically unstructured, with mixed formats, irrelevant sections, and varying quality. Techniques like Fusion Retrieval, Contextual Compression, and Reranking address noise and irrelevance through hybrid search, filtering, and refinement.
- **Google-Like Query Compatibility**: Google searches are diverse (keyword-driven, intent-based, or complex). Query Transformations and Fusion Retrieval accommodate this diversity by refining queries and combining keyword/semantic search. CRAG’s web fallback further aligns with Google’s broad sourcing.
- **Efficiency and Precision**: Contextual Compression and Reranking optimize the context window and retrieval precision, ensuring fast, relevant responses akin to Google’s user expectations.

---

### Less Optimal Techniques for This Scenario

While the above techniques are highly suitable, other RAG methods from the provided documents may be less effective for unstructured web-scraped content with Google-like queries due to specific limitations:

1. **Hierarchical Indices (18_hierarchy_rag.md)**: Relies on structured document sections and summaries, which may not exist in unstructured web content. Better for organized datasets like legal or academic documents.
2. **Proposition Chunking (14_proposition_chunking.md)**: Focuses on atomic factual statements, which may be hard to extract from noisy web data. Suited for fact-based retrieval in structured domains.
3. **Hypothetical Document Embedding (HyDE) (19_HyDE_rag.md)**: Effective for complex queries but may overcomplicate simple Google-like keyword searches, adding unnecessary overhead.
4. **Feedback Loop RAG (11_feedback_loop_rag.md)**: Requires user feedback for improvement, which may not be feasible in real-time web search scenarios. Better for iterative systems like chatbots.
5. **Simple RAG (1_simple_rag.md)**: Too basic for noisy web data, lacking advanced noise handling or query refinement. Best for prototyping or small-scale applications.
6. **Simple RAG with RL (21_rag_with_rl.md)**: RL optimization is computationally intensive and requires training, making it impractical for real-time web searches. Suited for research or high-accuracy domains.
7. **Graph RAG (17_graph_rag.md)**: Depends on structured relationships, which are challenging to construct from unstructured web content. Ideal for relational queries in organized data.
8. **Context-Enriched RAG (4_context_enriched_rag.md)**: Assumes neighboring chunks provide coherence, which may not hold in fragmented web data. Better for narrative-driven documents.
9. **Adaptive Retrieval (12_adaptive_rag.md)**: Query classification is powerful but may be overkill for typical Google searches, which often don’t require deep categorization. Suited for diverse enterprise queries.
10. **Chunk Size Selector (3_chunk_size_selector.md)**: Focused on tuning chunk sizes, not directly addressing web data noise or query diversity. Best for optimization experiments.
11. **Self-RAG (13_self_rag.md)**: Dynamic retrieval decisions add complexity, potentially slowing down responses for fast Google-like searches. Better for reliable, critical applications.
12. **Relevant Segment Extraction (RSE) (9_rse.md)**: Assumes clustered relevant chunks, which may not apply to disjointed web content. Suited for coherent document analysis.
13. **Multi-Modal RAG (15_multimodel_rag.md)**: Requires images, which may not be prevalent or relevant in all web-scraped content. Ideal for visual-heavy documents.
14. **Semantic Chunking (2_semantic_chunking.md)**: Content-based splitting is useful but may struggle with highly unstructured web data lacking clear semantic boundaries. Better for research or prototyping.
15. **Document Augmentation RAG (6_doc_augmentation_rag.md)**: Question generation enhances retrieval but adds processing overhead, less critical for keyword-driven Google searches. Suited for FAQ systems.
16. **Contextual Chunk Headers (5_contextual_chunk_headers_rag.md)**: Relies on structured headers, which are often absent in web-scraped content. Best for documents with clear hierarchies.
17. **Learning RAG Configurations (best_rag_finder.md)**: Focused on experimentation and tuning, not real-time search. Ideal for developers optimizing RAG systems.

---

### Implementation Considerations

To effectively apply the recommended techniques:

- **Preprocessing**: Clean web-scraped content to remove boilerplate (e.g., ads, footers) before processing. Techniques like Contextual Compression can further filter noise post-retrieval.
- **Query Handling**: Use Query Transformations to preprocess Google-like queries, ensuring they are refined or decomposed for optimal retrieval.
- **Hybrid Retrieval**: Implement Fusion Retrieval as the core retrieval mechanism, balancing keyword and semantic search to match Google’s versatility.
- **Refinement**: Apply Reranking to fine-tune retrieved results and Contextual Compression to focus the context, ensuring concise, relevant responses.
- **Fallback Mechanism**: Incorporate CRAG’s web search fallback for queries where scraped content is insufficient, mimicking Google’s ability to source external data.
- **Evaluation**: Continuously evaluate retrieval and response quality using metrics like relevance and faithfulness (as in Learning RAG Configurations) to ensure alignment with user expectations.

---

### Conclusion

For searching unstructured web-scraped content with Google-like queries, **Fusion Retrieval**, **Query Transformations**, **Contextual Compression**, **Reranking**, and **Corrective RAG (CRAG)** are the most effective techniques. They address the challenges of noisy data, diverse query types, and the need for precise, relevant responses. Fusion Retrieval’s hybrid approach and Query Transformations’ query refinement align closely with Google’s search paradigm, while Contextual Compression and Reranking ensure high-quality context. CRAG’s web fallback adds robustness for incomplete datasets. Other techniques, while valuable in specific contexts, are less suited due to their reliance on structured data, computational complexity, or mismatch with typical web search needs.
"""
logger.info("### Recommended RAG Techniques for Unstructured Web-Scraped Content with Google-Like Queries")

logger.info("\n\n[DONE]", bright=True)