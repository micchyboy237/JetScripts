async def main():
    from jet.transformers.formatters import format_json
    from dotenv import load_dotenv, find_dotenv
    from jet.logger import CustomLogger
    from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
    import asyncio
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/deepinfra.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # DeepInfra
    
    With this integration, you can use the DeepInfra embeddings model to get embeddings for your text data. Here is the link to the [embeddings models](https://deepinfra.com/models/embeddings).
    
    First, you need to sign up on the [DeepInfra website](https://deepinfra.com/) and get the API token. You can copy `model_ids` from the model cards and start using them in your code.
    
    ### Installation
    """
    logger.info("# DeepInfra")
    
    # !pip install llama-index llama-index-embeddings-deepinfra
    
    """
    ### Initialization
    """
    logger.info("### Initialization")
    
    
    _ = load_dotenv(find_dotenv())
    
    model = DeepInfraEmbeddingModel(
        model_id="BAAI/bge-large-en-v1.5",  # Use custom model ID
        api_token="YOUR_API_TOKEN",  # Optionally provide token here
        normalize=True,  # Optional normalization
        text_prefix="text: ",  # Optional text prefix
        query_prefix="query: ",  # Optional query prefix
    )
    
    """
    ### Synchronous Requests
    
    #### Get Text Embedding
    """
    logger.info("### Synchronous Requests")
    
    response = model.get_text_embedding("hello world")
    logger.debug(response)
    
    """
    #### Batch Requests
    """
    logger.info("#### Batch Requests")
    
    texts = ["hello world", "goodbye world"]
    response_batch = model.get_text_embedding_batch(texts)
    logger.debug(response_batch)
    
    """
    #### Query Requests
    """
    logger.info("#### Query Requests")
    
    query_response = model.get_query_embedding("hello world")
    logger.debug(query_response)
    
    """
    ### Asynchronous Requests
    
    #### Get Text Embedding
    """
    logger.info("### Asynchronous Requests")
    
    async def main():
        text = "hello world"
        async_response = await model.aget_text_embedding(text)
        logger.success(format_json(async_response))
        logger.success(format_json(async_response))
        logger.debug(async_response)
    
    
    if __name__ == "__main__":
    
        asyncio.run(main())
    
    """
    ---
    
    For any questions or feedback, please contact us at feedback@deepinfra.com.
    """
    logger.info("For any questions or feedback, please contact us at feedback@deepinfra.com.")
    
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