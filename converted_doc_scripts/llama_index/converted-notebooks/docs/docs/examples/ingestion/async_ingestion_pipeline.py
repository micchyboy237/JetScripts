async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.extractors import TitleExtractor, SummaryExtractor
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import MetadataMode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
    # Async Ingestion Pipeline + Metadata Extraction
    
    Recently, LlamaIndex has introduced async metadata extraction. Let's compare metadata extraction speeds in an ingestion pipeline using a newer and older version of LlamaIndex.
    
    We will test a pipeline using the classic Paul Graham essay.
    """
    logger.info("# Async Ingestion Pipeline + Metadata Extraction")

    # %pip install llama-index-embeddings-huggingface
    # %pip install llama-index-llms-ollama

    # !mkdir -p 'data/paul_graham/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


    # os.environ["OPENAI_API_KEY"] = "sk-..."

    """
    ## New LlamaIndex Ingestion
    
    Using a version of LlamaIndex greater or equal to v0.9.7, we can take advantage of improved async metadata extraction within ingestion pipelines.
    
    **NOTE:** Restart your notebook after installing a new version!
    """
    logger.info("## New LlamaIndex Ingestion")

    # !pip install "llama_index>=0.9.7"

    """
    **NOTE:** The `num_workers` kwarg controls how many requests can be outgoing at a given time using an async semaphore. Setting it higher may increase speeds, but can also lead to timeouts or rate limits, so set it wisely.
    """



    def build_pipeline():
        llm = OllamaFunctionCalling(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.1)

        transformations = [
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            TitleExtractor(
                llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8
            ),
            SummaryExtractor(
                llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8
            ),
            HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
        ]

        return IngestionPipeline(transformations=transformations)


    documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


    times = []
    for _ in range(3):
        time.sleep(30)  # help prevent rate-limits/timeouts, keeps each run fair
        pipline = build_pipeline()
        start = time.time()
        nodes = await pipline.arun(documents=documents)
        logger.success(format_json(nodes))
        end = time.time()
        times.append(end - start)

    logger.debug(f"Average time: {sum(times) / len(times)}")

    """
    The current `openai` python client package is a tad unstable -- sometimes async jobs will timeout, skewing the average. You can see the last progress bar took 1 minute instead of the previous 6 or 7 seconds, skewing the average.
    
    ## Old LlamaIndex Ingestion
    
    Now, lets compare to an older version of LlamaIndex, which was using "fake" async for metadata extraction.
    
    **NOTE:** Restart your notebook after installing the new version!
    """
    logger.info("## Old LlamaIndex Ingestion")

    # !pip install "llama_index<0.9.6"


    # os.environ["OPENAI_API_KEY"] = "sk-..."



    def build_pipeline():
        llm = OllamaFunctionCalling(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.1)

        transformations = [
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            TitleExtractor(llm=llm, metadata_mode=MetadataMode.EMBED),
            SummaryExtractor(llm=llm, metadata_mode=MetadataMode.EMBED),
            HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
        ]

        return IngestionPipeline(transformations=transformations)


    documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


    times = []
    for _ in range(3):
        time.sleep(30)  # help prevent rate-limits/timeouts, keeps each run fair
        pipline = build_pipeline()
        start = time.time()
        nodes = await pipline.arun(documents=documents)
        logger.success(format_json(nodes))
        end = time.time()
        times.append(end - start)

    logger.debug(f"Average time: {sum(times) / len(times)}")

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