async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.core import Document
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.extractors import TitleExtractor
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from pstats import SortKey
    import asyncio
    import cProfile, pstats
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/ingestion/parallel_execution_ingestion_pipeline.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Parallelizing Ingestion Pipeline
    
    In this notebook, we demonstrate how to execute ingestion pipelines using parallel processes. Both sync and async versions of batched parallel execution are possible with `IngestionPipeline`.
    """
    logger.info("# Parallelizing Ingestion Pipeline")
    
    # %pip install llama-index-embeddings-huggingface
    
    # import nest_asyncio
    
    # nest_asyncio.apply()
    
    
    """
    ### Load data
    
    For this notebook, we'll load the `PatronusAIFinanceBenchDataset` llama-dataset from [llamahub](https://llamahub.ai).
    """
    logger.info("### Load data")
    
    # !llamaindex-cli download-llamadataset PatronusAIFinanceBenchDataset --download-dir ./data
    
    
    documents = SimpleDirectoryReader(input_dir="./data/source_files").load_data()
    
    """
    ### Define our IngestionPipeline
    """
    logger.info("### Define our IngestionPipeline")
    
    
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            TitleExtractor(),
            HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
        ]
    )
    
    pipeline.disable_cache = True
    
    """
    ### Parallel Execution
    
    A single run. Setting `num_workers` to a value greater than 1 will invoke parallel execution.
    """
    logger.info("### Parallel Execution")
    
    nodes = pipeline.run(documents=documents, num_workers=4)
    
    len(nodes)
    
    # %timeit pipeline.run(documents=documents, num_workers=4)
    
    cProfile.run(
        "pipeline.run(documents=documents, num_workers=4)",
        "newstats",
    )
    p = pstats.Stats("newstats")
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(15)
    
    """
    ### Async Parallel Execution
    
    Here the `ProcessPoolExecutor` from `concurrent.futures` is used to execute processes asynchronously. The tasks are being processed are blocking, but also performed asynchronously on the individual processes.
    """
    logger.info("### Async Parallel Execution")
    
    nodes = await pipeline.arun(documents=documents, num_workers=4)
    logger.success(format_json(nodes))
    
    len(nodes)
    
    
    loop = asyncio.get_event_loop()
    # %timeit loop.run_until_complete(pipeline.arun(documents=documents, num_workers=4))
    
    loop = asyncio.get_event_loop()
    cProfile.run(
        "loop.run_until_complete(pipeline.arun(documents=documents, num_workers=4))",
        "async-newstats",
    )
    p = pstats.Stats("async-newstats")
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(15)
    
    """
    ### Sequential Execution
    
    By default `num_workers` is set to `None` and this will invoke sequential execution.
    """
    logger.info("### Sequential Execution")
    
    nodes = pipeline.run(documents=documents)
    
    len(nodes)
    
    # %timeit pipeline.run(documents=documents)
    
    cProfile.run("pipeline.run(documents=documents)", "oldstats")
    p = pstats.Stats("oldstats")
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(15)
    
    """
    ### Async on Main Processor
    
    As with the sync case, `num_workers` is default to `None`, which will then lead to single-batch execution of async tasks.
    """
    logger.info("### Async on Main Processor")
    
    nodes = await pipeline.arun(documents=documents)
    logger.success(format_json(nodes))
    
    len(nodes)
    
    # %timeit loop.run_until_complete(pipeline.arun(documents=documents))
    
    cProfile.run(
        "loop.run_until_complete(pipeline.arun(documents=documents))",
        "async-oldstats",
    )
    p = pstats.Stats("async-oldstats")
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(15)
    
    """
    ### In Summary
    
    The results from the above experiments are re-shared below where each strategy is listed from fastest to slowest with this example dataset and pipeline.
    
    1. (Async, Parallel Processing): 20.3s 
    2. (Async, No Parallel Processing): 20.5s
    3. (Sync, Parallel Processing): 29s
    4. (Sync, No Parallel Processing): 1min 11s
    
    We can see that both cases that use Parallel Processing outperforms the Sync, No Parallel Processing (i.e., `.run(num_workers=None)`). Also, that at least for this case for Async tasks, there is little gains in using Parallel Processing. Perhaps for larger workloads and IngestionPipelines, using Async with Parallel Processing can lead to larger gains.
    """
    logger.info("### In Summary")
    
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