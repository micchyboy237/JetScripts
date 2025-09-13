from jet._token.token_utils import token_counter


async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core import SimpleDirectoryReader, Document
    from llama_index.core import SummaryIndex
    from llama_index.core.evaluation import CorrectnessEvaluator
    from llama_index.core.utils import globals_helper
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Stress-Testing Long Context LLMs with a Recall Task
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_retrieval_benchmark.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    In this section we stress-test long context recall capabilities of GPT-4 and Claude v2. This is inspired by [Greg Kamradt's tweet](https://x.com/GregKamradt/status/1722386725635580292?s=20). 
    
    Similarly, we analyze the "needle in a haystack" recall capabilities of long-context LLms. We show an incremental extension by 1) adding Claude, and 2) testing recall where context **exceeds** context window, triggering response synthesis strategies.
    
    We use a fixed document - the 2021 Uber 10-K, which contains ~290k tokens.
    """
    logger.info("# Stress-Testing Long Context LLMs with a Recall Task")

    # %pip install llama-index-llms-ollama
    # %pip install llama-index-llms-anthropic

    # import nest_asyncio

    # nest_asyncio.apply()

    """
    ## Setup Data / Indexes
    
    We load the Uber 10-k
    """
    logger.info("## Setup Data / Indexes")

    # !mkdir -p 'data/10k/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'

    uber_docs0 = SimpleDirectoryReader(
        input_files=[
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10k/uber_2021.pdf"]
    ).load_data()
    uber_doc = Document(text="\n\n".join(
        [d.get_content() for d in uber_docs0]))

    """
    We print the number of tokens below. Note that this overflows the context window of existing LLMs, requiring response synthesis strategies.
    """
    logger.info("We print the number of tokens below. Note that this overflows the context window of existing LLMs, requiring response synthesis strategies.")

    num_tokens = token_counter(uber_doc.get_content(), "llama3.2")
    logger.debug(f"NUM TOKENS: {num_tokens}")

    """
    ## Try Out Different Experiments
    
    ### Define Context String
    
    Here we insert a single sentence of context that we're going to "hide" within the overall document at different positions.
    """
    logger.info("## Try Out Different Experiments")

    context_str = "Jerry's favorite snack is Hot Cheetos."
    query_str = "What is Jerry's favorite snack?"

    def augment_doc(doc_str, context, position):
        """Augment doc with additional context at a given position."""
        doc_str1 = doc_str[:position]
        doc_str2 = doc_str[position:]

        return f"{doc_str1}...\n\n{context}\n\n...{doc_str2}"

    test_str = augment_doc(
        uber_doc.get_content(), context_str, int(0.5 * len(uber_doc.get_content()))
    )

    """
    ### Define Experiment Loop
    
    The experiment loop is the following:
    1. Go through the set of positions (indicated by a percentile relative to the length of the doc)
    2. For each position, inject the context string at that position.
    3. Load the entire doc into our `SummaryIndex`, get the corresponding query engine.
    4. When a question is asked, we trigger response synthesis over the entire document (create-and-refine, or tree summarize).
    5. Compare predicted response against expected response with our `CorrectnessEvaluator`
    """
    logger.info("### Define Experiment Loop")

    async def run_experiments(
        doc, position_percentiles, context_str, query, llm, response_mode="compact"
    ):
        eval_llm = OllamaFunctionCalling(model="llama3.2")

        correctness_evaluator = CorrectnessEvaluator(llm=eval_llm)
        eval_scores = {}
        for idx, position_percentile in enumerate(position_percentiles):
            logger.debug(f"Position percentile: {position_percentile}")
            position_idx = int(position_percentile *
                               len(uber_doc.get_content()))
            new_doc_str = augment_doc(
                uber_doc.get_content(), context_str, position_idx
            )
            new_doc = Document(text=new_doc_str)
            index = SummaryIndex.from_documents(
                [new_doc],
            )
            query_engine = index.as_query_engine(
                response_mode=response_mode, llm=llm
            )
            logger.debug(f"Query: {query}")

            response = query_engine.query(query)
            logger.debug(f"Response: {str(response)}")
            eval_result = correctness_evaluator.evaluate(
                query=query, response=str(response), reference=context_str
            )
            eval_score = eval_result.score
            logger.debug(f"Eval score: {eval_score}")
            eval_scores[position_percentile] = eval_score
        return eval_scores

    position_percentiles = [0.0, 0.1, 0.2,
                            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    llm = OllamaFunctionCalling(model="llama3.2")

    eval_scores_gpt4 = await run_experiments(
        [uber_doc],
        position_percentiles,
        context_str,
        query_str,
        llm,
        response_mode="compact",
    )
    logger.success(format_json(eval_scores_gpt4))

    llm = OllamaFunctionCalling(model="llama3.2")
    eval_scores_gpt4_ts = await run_experiments(
        [uber_doc],
        position_percentiles,
        context_str,
        query_str,
        llm,
        response_mode="tree_summarize",
    )
    logger.success(format_json(eval_scores_gpt4_ts))

    llm = OllamaFunctionCalling(model="llama3.2")

    eval_scores_anthropic = await run_experiments(
        [uber_doc], position_percentiles, context_str, query_str, llm
    )
    logger.success(format_json(eval_scores_anthropic))

    llm = OllamaFunctionCalling(model="llama3.2")
    eval_scores_anthropic = await run_experiments(
        [uber_doc],
        position_percentiles,
        context_str,
        query_str,
        llm,
        response_mode="tree_summarize",
    )
    logger.success(format_json(eval_scores_anthropic))

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
