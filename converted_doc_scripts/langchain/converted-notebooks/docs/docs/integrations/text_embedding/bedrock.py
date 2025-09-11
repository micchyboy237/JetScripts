from jet.logger import logger
from langchain_aws import BedrockEmbeddings
import os
import shutil

async def main():
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger.basicConfig(filename=log_file)
    logger.info(f"Logs: {log_file}")
    
    PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    """
    # Bedrock
    
    >[Amazon Bedrock](https://aws.amazon.com/bedrock/) is a fully managed service that offers a choice of 
    > high-performing foundation models (FMs) from leading AI companies like `AI21 Labs`, `Ollama`, `Cohere`, 
    > `Meta`, `Stability AI`, and `Amazon` via a single API, along with a broad set of capabilities you need to 
    > build generative AI applications with security, privacy, and responsible AI. Using `Amazon Bedrock`, 
    > you can easily experiment with and evaluate top FMs for your use case, privately customize them with 
    > your data using techniques such as fine-tuning and `Retrieval Augmented Generation` (`RAG`), and build 
    > agents that execute tasks using your enterprise systems and data sources. Since `Amazon Bedrock` is 
    > serverless, you don't have to manage any infrastructure, and you can securely integrate and deploy 
    > generative AI capabilities into your applications using the AWS services you are already familiar with.
    """
    logger.info("# Bedrock")
    
    # %pip install --upgrade --quiet  boto3
    
    
    embeddings = BedrockEmbeddings(
        credentials_profile_name="bedrock-admin", region_name="us-east-1"
    )
    
    embeddings.embed_query("This is a content of the document")
    
    embeddings.embed_documents(
        ["This is a content of the document", "This is another document"]
    )
    
    await embeddings.aembed_query("This is a content of the document")
    
    await embeddings.aembed_documents(
        ["This is a content of the document", "This is another document"]
    )
    
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