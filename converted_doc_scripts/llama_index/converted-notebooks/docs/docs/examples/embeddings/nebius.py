async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.embeddings.nebius import NebiusEmbedding
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/nebius.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Nebius Embeddings
    
    This notebook demonstrates how to use [Nebius AI Studio](https://studio.nebius.ai/) Embeddings with LlamaIndex. Nebius AI Studio implements all state-of-the-art embeddings models, available for commercial use.
    
    First, let's install LlamaIndex and dependencies of Nebius AI Studio.
    """
    logger.info("# Nebius Embeddings")
    
    # %pip install llama-index-embeddings-nebius llama-index
    
    """
    Upload your Nebius AI Studio key from system variables below or simply insert it. You can get it by registering for free at [Nebius AI Studio](https://auth.eu.nebius.com/ui/login) and issuing the key at [API Keys section](https://studio.nebius.ai/settings/api-keys).
    """
    logger.info("Upload your Nebius AI Studio key from system variables below or simply insert it. You can get it by registering for free at [Nebius AI Studio](https://auth.eu.nebius.com/ui/login) and issuing the key at [API Keys section](https://studio.nebius.ai/settings/api-keys).")
    
    
    NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")  # NEBIUS_API_KEY = ""
    
    """
    Now let's get embeddings using Nebius AI Studio
    """
    logger.info("Now let's get embeddings using Nebius AI Studio")
    
    
    embed_model = NebiusEmbedding(api_key=NEBIUS_API_KEY)
    
    """
    ### Basic usage
    """
    logger.info("### Basic usage")
    
    text = "Everyone loves justice at another person's expense"
    embeddings = embed_model.get_text_embedding(text)
    assert len(embeddings) == 4096
    logger.debug(len(embeddings), embeddings[:5], sep="\n")
    
    """
    ### Asynchronous usage
    """
    logger.info("### Asynchronous usage")
    
    text = "Everyone loves justice at another person's expense"
    embeddings = await embed_model.aget_text_embedding(text)
    logger.success(format_json(embeddings))
    assert len(embeddings) == 4096
    logger.debug(len(embeddings), embeddings[:5], sep="\n")
    
    """
    ### Batched usage
    """
    logger.info("### Batched usage")
    
    texts = [
        "As the hours pass",
        "I will let you know",
        "That I need to ask",
        "Before I'm alone",
    ]
    
    embeddings = embed_model.get_text_embedding_batch(texts)
    assert len(embeddings) == 4
    assert len(embeddings[0]) == 4096
    logger.debug(*[x[:3] for x in embeddings], sep="\n")
    
    """
    ### Async batched usage
    """
    logger.info("### Async batched usage")
    
    texts = [
        "As the hours pass",
        "I will let you know",
        "That I need to ask",
        "Before I'm alone",
    ]
    
    embeddings = await embed_model.aget_text_embedding_batch(texts)
    logger.success(format_json(embeddings))
    assert len(embeddings) == 4
    assert len(embeddings[0]) == 4096
    logger.debug(*[x[:3] for x in embeddings], sep="\n")
    
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