async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import VectorStoreIndex
    from llama_index.core.evaluation import (
        CorrectnessEvaluator,
        SemanticSimilarityEvaluator,
    )
    from llama_index.core.evaluation import BatchEvalRunner
    from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
    from llama_index.core.evaluation.eval_utils import (
        get_responses,
        get_results_df,
    )
    from llama_index.core.extractors import (
        TitleExtractor,
        QuestionsAnsweredExtractor,
        SummaryExtractor,
    )
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import HTMLNodeParser, SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.readers.file import FlatReader
    from pathlib import Path
    import os
    import pandas as pd
    import pickle
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Transforms Evaluation
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/transforms/TransformsEval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    Here we try out different transformations and evaluate their quality.
    - First we try out different parsers (PDF, JSON)
    - Then we try out different extractors
    """
    logger.info("# Transforms Evaluation")

    # %pip install llama-index-readers-file
    # %pip install llama-index-llms-ollama
    # %pip install llama-index-embeddings-huggingface

    # !pip install llama-index

    """
    ## Load Data + Setup
    
    Load in the Tesla data.
    """
    logger.info("## Load Data + Setup")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # !wget "https://www.dropbox.com/scl/fi/mlaymdy1ni1ovyeykhhuk/tesla_2021_10k.htm?rlkey=qf9k4zn0ejrbm716j0gg7r802&dl=1" -O tesla_2021_10k.htm
    # !wget "https://www.dropbox.com/scl/fi/rkw0u959yb4w8vlzz76sa/tesla_2020_10k.htm?rlkey=tfkdshswpoupav5tqigwz1mp7&dl=1" -O tesla_2020_10k.htm

    reader = FlatReader()
    docs = reader.load_data(Path("./tesla_2020_10k.htm"))

    """
    ## Generate Eval Dataset / Define Eval Functions
    
    Generate a "golden" eval dataset from the Tesla documents.
    
    Also define eval functions for running a pipeline.
    
    Here we define an ingestion pipeline purely for generating a synthetic eval dataset.
    """
    logger.info("## Generate Eval Dataset / Define Eval Functions")

    # import nest_asyncio

    # nest_asyncio.apply()

    reader = FlatReader()
    docs = reader.load_data(Path("./tesla_2020_10k.htm"))

    pipeline = IngestionPipeline(
        documents=docs,
        transformations=[
            HTMLNodeParser.from_defaults(),
            SentenceSplitter(chunk_size=1024, chunk_overlap=200),
            HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
        ],
    )
    eval_nodes = pipeline.run(documents=docs)

    eval_llm = OllamaFunctionCallingAdapter(model="llama3.2")

    dataset_generator = DatasetGenerator(
        eval_nodes[:100],
        llm=eval_llm,
        show_progress=True,
        num_questions_per_chunk=3,
    )

    eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=100)
    logger.success(format_json(eval_dataset))

    len(eval_dataset.qr_pairs)

    eval_dataset.save_json("data/tesla10k_eval_dataset.json")

    eval_dataset = QueryResponseDataset.from_json(
        "data/tesla10k_eval_dataset.json"
    )

    eval_qs = eval_dataset.questions
    qr_pairs = eval_dataset.qr_pairs
    ref_response_strs = [r for (_, r) in qr_pairs]

    """
    ### Run Evals
    """
    logger.info("### Run Evals")

    evaluator_c = CorrectnessEvaluator(llm=eval_llm)
    evaluator_s = SemanticSimilarityEvaluator(llm=eval_llm)
    evaluator_dict = {
        "correctness": evaluator_c,
        "semantic_similarity": evaluator_s,
    }
    batch_eval_runner = BatchEvalRunner(
        evaluator_dict, workers=2, show_progress=True
    )

    async def run_evals(
        pipeline, batch_eval_runner, docs, eval_qs, eval_responses_ref
    ):
        nodes = pipeline.run(documents=docs)
        vector_index = VectorStoreIndex(nodes)
        query_engine = vector_index.as_query_engine()

        pred_responses = get_responses(
            eval_qs, query_engine, show_progress=True)
        eval_results = batch_eval_runner.evaluate_responses(
            eval_qs, responses=pred_responses, reference=eval_responses_ref
        )
        logger.success(format_json(eval_results))
        return eval_results

    """
    ## 1. Try out Different Sentence Splitter (Overlaps)
    
    The chunking strategy matters! Here we try the sentence splitter with different overlap values, to see how it impacts performance.
    
    The `IngestionPipeline` lets us concisely define an e2e transformation pipeline for RAG, and we define variants where each corresponds to a different sentence splitter configuration (while keeping other steps fixed).
    """
    logger.info("## 1. Try out Different Sentence Splitter (Overlaps)")

    sent_parser_o0 = SentenceSplitter(chunk_size=1024, chunk_overlap=0)
    sent_parser_o200 = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    sent_parser_o500 = SentenceSplitter(chunk_size=1024, chunk_overlap=600)

    html_parser = HTMLNodeParser.from_defaults()

    parser_dict = {
        "sent_parser_o0": sent_parser_o0,
        "sent_parser_o200": sent_parser_o200,
        "sent_parser_o500": sent_parser_o500,
    }

    """
    Define a separate pipeline for each parser.
    """
    logger.info("Define a separate pipeline for each parser.")

    pipeline_dict = {}
    for k, parser in parser_dict.items():
        pipeline = IngestionPipeline(
            documents=docs,
            transformations=[
                html_parser,
                parser,
                HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
            ],
        )
        pipeline_dict[k] = pipeline

    eval_results_dict = {}
    for k, pipeline in pipeline_dict.items():
        eval_results = await run_evals(
            pipeline, batch_eval_runner, docs, eval_qs, ref_response_strs
        )
        logger.success(format_json(eval_results))
        eval_results_dict[k] = eval_results

    pickle.dump(eval_results_dict, open("eval_results_1.pkl", "wb"))

    eval_results_list = list(eval_results_dict.items())

    results_df = get_results_df(
        [v for _, v in eval_results_list],
        [k for k, _ in eval_results_list],
        ["correctness", "semantic_similarity"],
    )
    display(results_df)

    for k, pipeline in pipeline_dict.items():
        pipeline.cache.persist(f"./cache/{k}.json")

    """
    ## 2. Try out Different Extractors
    
    Similarly, metadata extraction can be quite important for good performance. We experiment with this as a last step in an overall ingestion pipeline, and define different ingestion pipeline variants corresponding to different extractors.
    
    We define the set of document extractors we want to try out. 
    
    We keep the parsers fixed (HTML parser, sentence splitter w/ overlap 200) and the embedding model fixed (HuggingFaceEmbedding).
    """
    logger.info("## 2. Try out Different Extractors")

    extractor_dict = {
        "summary": SummaryExtractor(in_place=False),
        "qa": QuestionsAnsweredExtractor(in_place=False),
        "default": None,
    }

    html_parser = HTMLNodeParser.from_defaults()
    sent_parser_o200 = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    pipeline_dict = {}
    html_parser = HTMLNodeParser.from_defaults()
    for k, extractor in extractor_dict.items():
        if k == "default":
            transformations = [
                html_parser,
                sent_parser_o200,
                HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
            ]
        else:
            transformations = [
                html_parser,
                sent_parser_o200,
                extractor,
                HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
            ]

        pipeline = IngestionPipeline(transformations=transformations)
        pipeline_dict[k] = pipeline

    eval_results_dict_2 = {}
    for k, pipeline in pipeline_dict.items():
        eval_results = await run_evals(
            pipeline, batch_eval_runner, docs, eval_qs, ref_response_strs
        )
        logger.success(format_json(eval_results))
        eval_results_dict_2[k] = eval_results

    eval_results_list_2 = list(eval_results_dict_2.items())

    results_df = get_results_df(
        [v for _, v in eval_results_list_2],
        [k for k, _ in eval_results_list_2],
        ["correctness", "semantic_similarity"],
    )
    display(results_df)

    for k, pipeline in pipeline_dict.items():
        pipeline.cache.persist(f"./cache/{k}.json")

    """
    ## 3. Try out Multiple Extractors (with Caching)
    
    TODO
    
    Each extraction step can be expensive due to LLM calls. What if we want to experiment with multiple extractors? 
    
    We take advantage of **caching** so that all previous extractor calls are cached, and we only experiment with the final extractor call. The `IngestionPipeline` gives us a clean abstraction to play around with the final extractor.
    
    Try out different extractors
    """
    logger.info("## 3. Try out Multiple Extractors (with Caching)")

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
