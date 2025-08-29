async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.core.base.embeddings.base import SimilarityMode
    from llama_index.embeddings.vertex import VertexTextEmbedding
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # Vertex AI Text Embedding
    
    Imports the VertexTextEmbedding class and initializes an instance named embed_model with a specified project and location. Uses APPLICATION_DEFAULT_CREDENTIALS if no credentials is specified. The default model is `textembedding-gecko@003` in document retrival mode.
    """
    logger.info("# Vertex AI Text Embedding")
    
    
    embed_model = VertexTextEmbedding(project="speedy-atom-413006", location="us-central1")
    
    embed_model.dict()
    
    """
    ## Document and Query Retrival
    """
    logger.info("## Document and Query Retrival")
    
    embed_text_result = embed_model.get_text_embedding("Hello World!")
    
    embed_text_result[:5]
    
    embed_query_result = embed_model.get_query_embedding("Hello World!")
    
    embed_query_result[:5]
    
    
    embed_model.similarity(
        embed_text_result, embed_query_result, SimilarityMode.DOT_PRODUCT
    )
    
    """
    ## Using the async interface
    """
    logger.info("## Using the async interface")
    
    # import nest_asyncio
    
    # nest_asyncio.apply()
    
    result = await embed_model.aget_text_embedding("Hello World!")
    logger.success(format_json(result))
    
    result[:5]
    
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