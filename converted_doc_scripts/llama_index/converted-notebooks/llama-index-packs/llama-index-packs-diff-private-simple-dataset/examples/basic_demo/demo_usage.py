async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler
    from llama_index.core.llama_dataset.simple import LabelledSimpleDataset
    from llama_index.packs.diff_private_simple_dataset import DiffPrivateSimpleDatasetPack
    from llama_index.packs.diff_private_simple_dataset.base import PromptBundle
    import llama_index.core.instrumentation as instrument
    import os
    import shutil
    import tiktoken


    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Basic Usage: DiffPrivateSimpleDatasetPack
    
    In this notebook, we demonstrate the basic usage of `DiffPrivateSimpleDatasetPack`. The purpose of this pack is to create privacy, safe synthetic observations (or copies) of an original, likely sensitive dataset.
    
    ### How it works?
    `DiffPrivateSimpleDatasetPack` operates on the `LabelledSimpleDataset` type, which is a llama-dataset that contains `LabelledSimpleDataExample`'s for which there are two fields, namely: `text` and `reference_label`. Calling the `.run()` method of the `DiffPrivateSimpleDatasetPack` will ultimately produce a new `LabelledSimpleDataset` containing privacy-safe, synthetic `LabelledSimpleDataExample`'s.
    
    ### This notebook:
    In this notebook, we create a privacy-safe, synthetic version of the AGNEWs dataset. This raw AGNEWs data was used to create a `LabelledSimpleDataset` version from it (see `_create_agnews_simple_dataset.ipynb`.
    """
    logger.info("# Basic Usage: DiffPrivateSimpleDatasetPack")

    # %pip install treelib -q

    # import nest_asyncio

    # nest_asyncio.apply()


    span_handler = SimpleSpanHandler()
    dispatcher = instrument.get_dispatcher()
    dispatcher.add_span_handler(span_handler)


    """
    ### Load LabelledSimpleDataset
    """
    logger.info("### Load LabelledSimpleDataset")

    simple_dataset = LabelledSimpleDataset.from_json("./agnews.json")

    simple_dataset.to_pandas()[:5]

    simple_dataset.to_pandas().value_counts("reference_label")

    """
    ### InstantiatePack
    
    To construct a `DiffPrivateSimpleDatasetPack` object, we need to supply:
    
    1. an `LLM` (must return `CompletionResponse`),
    2. its associated `tokenizer`,
    3. a `PromptBundle` object that contains the parameters required for prompting the LLM to produce the synthetic observations
    4. a `LabelledSimpleDataset`
    5. [Optional] `sephamore_counter_size` used to help reduce chances of experiencing a `RateLimitError` when calling the LLM's completions API.
    6. [Optional] `sleep_time_in_seconds` used to help reduce chances of experiencing a `RateLimitError` when calling the LLM"s completions API.
    """
    logger.info("### InstantiatePack")

    llm = OllamaFunctionCalling(
        model="llama3.2", request_timeout=300.0, context_window=4096,
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,  # OllamaFunctionCalling only allows for top 5 next token as opposed to entire vocabulary
    )
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")

    prompt_bundle = PromptBundle(
        instruction=(
            "Given a label of news type, generate the chosen type of news accordingly.\n"
            "Start your answer directly after 'Text: '. Begin your answer with [RESULT].\n"
        ),
        label_heading="News Type",
        text_heading="Text",
    )

    dp_simple_dataset_pack = DiffPrivateSimpleDatasetPack(
        llm=llm,
        tokenizer=tokenizer,
        prompt_bundle=prompt_bundle,
        simple_dataset=simple_dataset,
    )

    """
    To generate a single synthetic example, we can call the `generate_dp_synthetic_example()` method. Synthetic examples are created for a specific `label`. Both sync and async are supported. A few params are required:
    
    - `label`: The class from which you want to generate a synthetic example.
    - `t_max`: The max number of tokens you would like to generate (the algorithm adds some logic per token in order to satisfy differential privacy).
    - `sigma`: Controls the variance of the noise distribution associated with differential privacy noise mechanism. A value of `sigma` amounts to a level of `epsilon` satisfied in differential privacy.
    - `num_splits`: The differential privacy algorithm implemented here relies on disjoint splits of the original dataset.
    - `num_samples_per_split`: The number of private, in-context examples to include in the generation of the synthetic example.
    """
    logger.info("To generate a single synthetic example, we can call the `generate_dp_synthetic_example()` method. Synthetic examples are created for a specific `label`. Both sync and async are supported. A few params are required:")

    dp_simple_dataset_pack.generate_dp_synthetic_example(
        label="Sports", t_max=35, sigma=0.1, num_splits=2, num_samples_per_split=8
    )

    await dp_simple_dataset_pack.agenerate_dp_synthetic_example(
        label="Sports", t_max=35, sigma=0.1, num_splits=2, num_samples_per_split=8
    )

    """
    To create a privacy-safe, synthetic dataset, we call the `run()` (or async `arun()`) method. The required params for this method have been priorly introduced, with the exception of `sizes`.
    
    - `sizes`: Can be `int` or `Dict[str, int]` which specify the number of synthetic observations per label to be generated.
    """
    logger.info("To create a privacy-safe, synthetic dataset, we call the `run()` (or async `arun()`) method. The required params for this method have been priorly introduced, with the exception of `sizes`.")

    synthetic_dataset = await dp_simple_dataset_pack.arun(
            sizes={"World": 1, "Sports": 1, "Sci/Tech": 0, "Business": 0},
            t_max=10,  #
            sigma=0.5,
            num_splits=2,
            num_samples_per_split=8,
        )
    logger.success(format_json(synthetic_dataset))

    synthetic_dataset.to_pandas()

    span_handler.print_trace_trees()

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