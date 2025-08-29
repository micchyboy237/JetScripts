async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
    )
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/text_embedding_inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Text Embedding Inference
    
    This notebook demonstrates how to configure `TextEmbeddingInference` embeddings.
    
    The first step is to deploy the embeddings server. For detailed instructions, see the [official repository for Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference). Or [tei-gaudi repository](https://github.com/huggingface/tei-gaudi) if you are deploying on Habana Gaudi/Gaudi 2. 
    
    Once deployed, the code below will connect to and submit embeddings for inference.
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Text Embedding Inference")
    
    # %pip install llama-index-embeddings-text-embeddings-inference
    
    # !pip install llama-index
    
    
    
    embed_model = TextEmbeddingsInference(
        model_name="BAAI/bge-large-en-v1.5",  # required for formatting inference text,
        timeout=60,  # timeout in seconds
        embed_batch_size=10,  # batch size for embedding
    )
    
    embeddings = embed_model.get_text_embedding("Hello World!")
    logger.debug(len(embeddings))
    logger.debug(embeddings[:5])
    
    embeddings = await embed_model.aget_text_embedding("Hello World!")
    logger.success(format_json(embeddings))
    logger.success(format_json(embeddings))
    logger.debug(len(embeddings))
    logger.debug(embeddings[:5])
    
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