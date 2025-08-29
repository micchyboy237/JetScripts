async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.embeddings.gemini import GeminiEmbedding
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/gemini.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Google Gemini Embeddings
    
    **NOTE:** This example is deprecated. Please use the `GoogleGenAIEmbedding` class instead, detailed [here](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/google_genai.ipynb).
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Google Gemini Embeddings")
    
    # %pip install llama-index-embeddings-gemini
    
    # !pip install llama-index 'google-generativeai>=0.3.0' matplotlib
    
    
    GOOGLE_API_KEY = ""  # add your GOOGLE API key here
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    
    model_name = "models/embedding-001"
    
    embed_model = GeminiEmbedding(
        model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document"
    )
    
    embeddings = embed_model.get_text_embedding("Google Gemini Embeddings.")
    
    logger.debug(f"Dimension of embeddings: {len(embeddings)}")
    
    embeddings[:5]
    
    embeddings = embed_model.get_query_embedding("Google Gemini Embeddings.")
    embeddings[:5]
    
    embeddings = embed_model.get_text_embedding(
        ["Google Gemini Embeddings.", "Google is awesome."]
    )
    
    logger.debug(f"Dimension of embeddings: {len(embeddings)}")
    logger.debug(embeddings[0][:5])
    logger.debug(embeddings[1][:5])
    
    embedding = await embed_model.aget_text_embedding("Google Gemini Embeddings.")
    logger.success(format_json(embedding))
    logger.success(format_json(embedding))
    logger.debug(embedding[:5])
    
    embeddings = await embed_model.aget_text_embedding_batch(
            [
                "Google Gemini Embeddings.",
                "Google is awesome.",
                "Llamaindex is awesome.",
            ]
        )
    logger.success(format_json(embeddings))
    logger.debug(embeddings[0][:5])
    logger.debug(embeddings[1][:5])
    logger.debug(embeddings[2][:5])
    
    embedding = await embed_model.aget_query_embedding("Google Gemini Embeddings.")
    logger.success(format_json(embedding))
    logger.success(format_json(embedding))
    logger.debug(embedding[:5])
    
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