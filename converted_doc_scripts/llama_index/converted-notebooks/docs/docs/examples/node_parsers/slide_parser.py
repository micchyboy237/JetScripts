async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import Document
    from llama_index.core.utilities.token_counting import TokenCounter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.node_parser.slide import SlideNodeParser
    import os
    import shutil
    import time
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # SlideNodeParser
    
    [SLIDE](https://arxiv.org/abs/2503.17952) (Sliding Localized Information for Document Extraction) is a chunking method introduced to enhance entity and relationship extraction in long documents, especially for low-resource languages. It was designed to support GraphRAG pipelines by embedding localized context into each chunk without exceeding the context window of the LLM.
    
    `SlideNodeParser` faithfully implements a almost similar version of this method to improve downstream retrieval and reasoning quality by generating short, meaningful context using a sliding window of nearby chunks. This technique is proven useful for Graph based Retrieval Augmented Generaration techniques.
    
    Here is the technique as outlined in the paper:
    
    ```
    Given a document D and a list of base chunks (C1, C2, ..., Ck) segmented by sentence boundaries and token count, SLIDE builds local context for each chunk using a fixed-size sliding window of neighboring chunks. This context is appended to the chunk using an LLM-generated summary.
    
    The window size is a hyperparameter defined based on the model’s context length and compute budget. Each chunk Ci is enriched by including a few preceding and succeeding chunks (e.g., 5 on each side), resulting in a total of window_size + 1 inputs sent to the LLM.
    
    This process is repeated for every chunk in the document. The result is a collection of chunks embedded with richer, window-specific local context, which significantly improves the quality of knowledge graphs and search retrieval, especially in multilingual or resource-constrained settings.
    ```
    """
    logger.info("# SlideNodeParser")
    
    # %pip install llama-index-node-parser-slide
    
    """
    ## Install ipy widgets for progress bars (Optional)
    """
    logger.info("## Install ipy widgets for progress bars (Optional)")
    
    # %pip install ipywidgets
    
    """
    ## Setup Data
    
    Here we consider a sample text.
    """
    logger.info("## Setup Data")
    
    text = """Constructing accurate knowledge graphs from long texts and low-resource languages is challenging, as large language models (LLMs) experience degraded performance with longer input chunks.
    This problem is amplified in low-resource settings where data scarcity hinders accurate entity and relationship extraction.
    Contextual retrieval methods, while improving retrieval accuracy, struggle with long documents.
    They truncate critical information in texts exceeding maximum context lengths of LLMs, significantly limiting knowledge graph construction.
    We introduce SLIDE (Sliding Localized Information for Document Extraction), a chunking method that processes long documents by generating local context through overlapping windows.
    SLIDE ensures that essential contextual information is retained, enhancing knowledge graph extraction from documents exceeding LLM context limits.
    It significantly improves GraphRAG performance, achieving a 24% increase in entity extraction and a 39% improvement in relationship extraction for English.
    For Afrikaans, a low-resource language, SLIDE achieves a 49% increase in entity extraction and an 82% improvement in relationship extraction.
    Furthermore, it improves upon state-of-the-art in question-answering metrics such as comprehensiveness, diversity and empowerment, demonstrating its effectiveness in multilingual and resource-constrained settings.
    
    Since SLIDE enhances knowledge graph construction in GraphRAG systems through contextual chunking, we first discuss related work in GraphRAG and chunking, highlighting their strengths and limitations.
    This sets the stage for our approach, which builds on GraphRAG by using overlapping windows to improve entity and relationship extraction.
    2.1 GraphRAG and Knowledge Graphs.
    GraphRAG (Edge et al., 2024) is an advanced RAG framework that integrates knowledge graphs with large language models (LLMs) (Trajanoska et al., 2023) to enhance reasoning and contextual understanding.
    Unlike traditional RAG systems, GraphRAG builds a knowledge graph with entities as nodes and relationships as edges, enabling precise and context-rich responses by leveraging the graph’s structure (Edge et al., 2024; Wu et al., 2024).
    Large language models (LLMs), such as GPT-4, show reduced effectiveness in entity and relationship extraction as input chunk lengths increase, degrading accuracy for longer texts (Edge et al., 2024).
    They also struggle with relationship extraction in low-resource languages, limiting their applicability (Chen et al., 2024; Jinensibieke et al., 2024).
    Building upon this work, our approach further enhances knowledge graph extraction by incorporating localized context which improves entity and relationship extraction.
    2.2 Contextual Chunking.
    Recent work in RAG systems has explored advanced chunking techniques to enhance retrieval and knowledge graph construction.
    Günther et al. (2024) implemented late chunking, where entire documents are embedded to capture global context before splitting into chunks, improving retrieval by emphasizing document-level coherence.
    However, this focus on global embeddings is less suited for knowledge graph construction.
    Our method instead uses localized context from raw text to retain meaningful relationships for improved entity and relationship extraction.
    Wu et al. (2024) introduced a hybrid chunking approach for Medical Graph RAG, combining structural markers like paragraphs with semantic coherence to produce self-contained chunks.
    While effective, this approach relies on predefined boundaries.
    Our method extends this by generating contextual information from neighboring chunks, enhancing the completeness of knowledge graph construction.
    Contextual retrieval (Anthropic, 2024) improves accuracy but struggles with longer documents, as embedding each chunk with full document context is computationally expensive and truncates critical information with documents exceeding maximum context length of the model (Jiang et al., 2024; Li et al., 2024).
    Our overlapping window-based approach addresses these inefficiencies, improving performance in both retrieval and knowledge graph construction.
    """
    
    
    document = Document(text=text)
    
    """
    ## Setup LLM
    """
    logger.info("## Setup LLM")
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your OllamaFunctionCallingAdapter API key
    
    
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
    llm = OllamaFunctionCallingAdapter(model="llama3.2")
    
    
    token_counter = TokenCounter()
    token_count = token_counter.get_string_tokens(text)
    logger.debug(f"Token count: {token_count}")
    
    """
    ## Setup SlideNodeParser
    """
    logger.info("## Setup SlideNodeParser")
    
    
    parser = SlideNodeParser.from_defaults(
        chunk_size=200,
        window_size=5,
    )
    
    """
    ### Run the synchronous blocking function
    """
    logger.info("### Run the synchronous blocking function")
    
    
    start_time = time.time()
    nodes = parser.get_nodes_from_documents([document], show_progress=True)
    end_time = time.time()
    logger.debug(f"Time taken to parse: {end_time - start_time} seconds")
    
    """
    ## Lets inspect chunks
    """
    logger.info("## Lets inspect chunks")
    
    for i, node in enumerate(nodes):
        logger.debug(f"\n--- Chunk {i+1} ---")
        logger.debug("Text:", node.text)
        logger.debug("Local Context:", node.metadata.get("local_context"))
    
    """
    ### Lets run the asynchronous version with parallel LLM calls
    """
    logger.info("### Lets run the asynchronous version with parallel LLM calls")
    
    parser.llm_workers = 4
    start_time = time.time()
    nodes = await parser.aget_nodes_from_documents([document], show_progress=True)
    logger.success(format_json(nodes))
    logger.success(format_json(nodes))
    end_time = time.time()
    logger.debug(f"Time taken to parse: {end_time - start_time} seconds")
    
    """
    ## Lets inspect the chunks
    """
    logger.info("## Lets inspect the chunks")
    
    for i, node in enumerate(nodes):
        logger.debug(f"\n--- Chunk {i+1} ---")
        logger.debug("Text:", node.text)
        logger.debug("Local Context:", node.metadata.get("local_context"))
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())