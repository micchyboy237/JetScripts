async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from dotenv import load_dotenv
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core import StorageContext
    from llama_index.core import VectorStoreIndex
    from llama_index.core.extractors import (
        QuestionsAnsweredExtractor,
        TitleExtractor,
    )
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from pinecone import Pinecone, Index, ServerlessSpec
    import fitz
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/low_level/ingestion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Building Data Ingestion from Scratch
    
    In this tutorial, we show you how to build a data ingestion pipeline into a vector database.
    
    We use Pinecone as the vector database.
    
    We will show how to do the following:
    1. How to load in documents.
    2. How to use a text splitter to split documents.
    3. How to **manually** construct nodes from each text chunk.
    4. [Optional] Add metadata to each Node.
    5. How to generate embeddings for each text chunk.
    6. How to insert into a vector database.
    
    ## Pinecone
    
    You will need a [pinecone.io](https://www.pinecone.io/) api key for this tutorial. You can [sign up for free](https://app.pinecone.io/?sessionType=signup) to get a Starter account.
    
    If you create a Starter account, you can name your application anything you like.
    
    Once you have an account, navigate to 'API Keys' in the Pinecone console. You can use the default key or create a new one for this tutorial.
    
    Save your api key and its environment (`gcp_starter` for free accounts). You will need them below.
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Building Data Ingestion from Scratch")

    # %pip install llama-index-embeddings-huggingface
    # %pip install llama-index-vector-stores-pinecone
    # %pip install llama-index-llms-ollama

    # !pip install llama-index

    """
    ## OllamaFunctionCalling
    
    You will need an [OllamaFunctionCalling](https://openai.com/) api key for this tutorial. Login to your [platform.openai.com](https://platform.openai.com/) account, click on your profile picture in the upper right corner, and choose 'API Keys' from the menu. Create an API key for this tutorial and save it. You will need it below.
    
    ## Environment
    
    First we add our dependencies.
    """
    logger.info("## OllamaFunctionCalling")

    # !pip -q install python-dotenv pinecone-client llama-index pymupdf

    """
    #### Set Environment Variables
    
    We create a file for our environment variables. Do not commit this file or share it!
    
    Note: Google Colabs will let you create but not open a .env
    """
    logger.info("#### Set Environment Variables")

    dotenv_path = (
        "env"  # Google Colabs will not let you open a .env, but you can set
    )
    with open(dotenv_path, "w") as f:
        f.write('PINECONE_API_KEY="<your api key>"\n')
    #     f.write('OPENAI_API_KEY="<your api key>"\n')

    """
    Set your OllamaFunctionCalling api key, and Pinecone api key and environment in the file we created.
    """
    logger.info(
        "Set your OllamaFunctionCalling api key, and Pinecone api key and environment in the file we created.")

    load_dotenv(dotenv_path=dotenv_path)

    """
    ## Setup
    
    We build an empty Pinecone Index, and define the necessary LlamaIndex wrappers/abstractions so that we can start loading data into Pinecone.
    
    
    Note: Do not save your API keys in the code or add pinecone_env to your repo!
    """
    logger.info("## Setup")

    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)

    index_name = "llamaindex-rag-fs"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=1536,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    pinecone_index = pc.Index(index_name)

    pinecone_index.delete(deleteAll=True)

    """
    #### Create PineconeVectorStore
    
    Simple wrapper abstraction to use in LlamaIndex. Wrap in StorageContext so we can easily load in Nodes.
    """
    logger.info("#### Create PineconeVectorStore")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    """
    ## Build an Ingestion Pipeline from Scratch
    
    We show how to build an ingestion pipeline as mentioned in the introduction.
    
    Note that steps (2) and (3) can be handled via our `NodeParser` abstractions, which handle splitting and node creation.
    
    For the purposes of this tutorial, we show you how to create these objects manually.
    
    ### 1. Load Data
    """
    logger.info("## Build an Ingestion Pipeline from Scratch")

    # !mkdir data
    # !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

    file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/llama2.pdf"
    doc = fitz.open(file_path)

    """
    ### 2. Use a Text Splitter to Split Documents
    
    Here we import our `SentenceSplitter` to split document texts into smaller chunks, while preserving paragraphs/sentences as much as possible.
    """
    logger.info("### 2. Use a Text Splitter to Split Documents")

    text_parser = SentenceSplitter(
        chunk_size=1024,
    )

    text_chunks = []
    doc_idxs = []
    for doc_idx, page in enumerate(doc):
        page_text = page.get_text("text")
        cur_text_chunks = text_parser.split_text(page_text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    """
    ### 3. Manually Construct Nodes from Text Chunks
    
    We convert each chunk into a `TextNode` object, a low-level data abstraction in LlamaIndex that stores content but also allows defining metadata + relationships with other Nodes.
    
    We inject metadata from the document into each node.
    
    This essentially replicates logic in our `SentenceSplitter`.
    """
    logger.info("### 3. Manually Construct Nodes from Text Chunks")

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc_idx = doc_idxs[idx]
        src_page = doc[src_doc_idx]
        nodes.append(node)

    logger.debug(nodes[0].metadata)

    logger.debug(nodes[0].get_content(metadata_mode="all"))

    """
    ### [Optional] 4. Extract Metadata from each Node
    
    We extract metadata from each Node using our Metadata extractors.
    
    This will add more metadata to each Node.
    """
    logger.info("### [Optional] 4. Extract Metadata from each Node")

    llm = OllamaFunctionCalling(
        model="llama3.2")

    extractors = [
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
    ]

    pipeline = IngestionPipeline(
        transformations=extractors,
    )
    nodes = await pipeline.arun(nodes=nodes, in_place=False)
    logger.success(format_json(nodes))

    logger.debug(nodes[0].metadata)

    """
    ### 5. Generate Embeddings for each Node
    
    Generate document embeddings for each Node using our OllamaFunctionCalling embedding model (`text-embedding-ada-002`).
    
    Store these on the `embedding` property on each Node.
    """
    logger.info("### 5. Generate Embeddings for each Node")

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    """
    ### 6. Load Nodes into a Vector Store
    
    We now insert these nodes into our `PineconeVectorStore`.
    
    **NOTE**: We skip the VectorStoreIndex abstraction, which is a higher-level abstraction that handles ingestion as well. We use `VectorStoreIndex` in the next section to fast-track retrieval/querying.
    """
    logger.info("### 6. Load Nodes into a Vector Store")

    vector_store.add(nodes)

    """
    ## Retrieve and Query from the Vector Store
    
    Now that our ingestion is complete, we can retrieve/query this vector store.
    
    **NOTE**: We can use our high-level `VectorStoreIndex` abstraction here. See the next section to see how to define retrieval at a lower-level!
    """
    logger.info("## Retrieve and Query from the Vector Store")

    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine()

    query_str = "Can you tell me about the key concepts for safety finetuning"

    response = query_engine.query(query_str)

    logger.debug(str(response))

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
