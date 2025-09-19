async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index import VectorStoreIndex
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex
    from llama_index.core.llama_dataset.generator import RagDatasetGenerator
    from llama_index.llama_datasets import download_llama_datasets
    from llama_index.llama_pack import download_llama_pack
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llama_dataset/uploading_llama_dataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Contributing a LlamaDataset To LlamaHub
    
    `LlamaDataset`'s storage is managed through a git repository. To contribute a dataset requires making a pull request to `llama_index/llama_datasets` Github (LFS) repository. 
    
    To contribute a `LabelledRagDataset` (a subclass of `BaseLlamaDataset`), two sets of files are required:
    
    1. The `LabelledRagDataset` saved as json named `rag_dataset.json`
    2. Source document files used to create the `LabelledRagDataset`
    
    This brief notebook provides a quick example using the Paul Graham Essay text file.
    """
    logger.info("# Contributing a LlamaDataset To LlamaHub")

    # %pip install llama-index-llms-ollama

    # import nest_asyncio

    # nest_asyncio.apply()

    """
    ### Load Data
    """
    logger.info("### Load Data")

    # !mkdir -p 'data/paul_graham/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

    documents = SimpleDirectoryReader(
        input_files=["data/paul_graham/paul_graham_essay.txt"]
    ).load_data()

    llm_gpt35 = OllamaFunctionCalling(
        model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.3)

    dataset_generator = RagDatasetGenerator.from_documents(
        documents,
        llm=llm_gpt35,
        num_questions_per_chunk=2,  # set the number of questions per nodes
        show_progress=True,
    )

    rag_dataset = dataset_generator.generate_dataset_from_nodes()

    """
    Now that we have our `LabelledRagDataset` generated (btw, it's totally fine to manually create one with human generated queries and reference answers!), we store this into the necessary json file.
    """
    logger.info("Now that we have our `LabelledRagDataset` generated (btw, it's totally fine to manually create one with human generated queries and reference answers!), we store this into the necessary json file.")

    rag_dataset.save_json("rag_dataset.json")

    """
    #### Generating Baseline Results
    
    In addition to adding just a `LlamaDataset`, we also encourage adding baseline benchmarks for others to use as sort of measuring stick against their own RAG pipelines.
    """
    logger.info("#### Generating Baseline Results")

    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()

    prediction_dataset = await rag_dataset.amake_predictions_with(
        query_engine=query_engine, show_progress=True
    )
    logger.success(format_json(prediction_dataset))

    """
    ## Submitting The Pull-Requests
    
    With the `rag_dataset.json` and source file `paul_graham_essay.txt` (note in this case, there is only one source document, but there can be several), we can perform the two steps for contributing a `LlamaDataset` into `LlamaHub`:
    
    1. Similar, to how contributions are made for `loader`'s, `agent`'s and `pack`'s, create a pull-request for `llama_hub` repository that adds a new folder for new `LlamaDataset`. This step uploads the information about the new `LlamaDataset` so that it can be presented in the `LlamaHub` UI.
    
    2. Create a pull request into `llama_datasets` repository to actually upload the data files.
    
    ### Step 0 (Pre-requisites)
    
    Fork and then clone (onto your local machine) both, the `llama_hub` Github repository as well as the `llama_datasets` one. You'll be submitting a pull requests into both of these repos from a new branch off of your forked versions.
    
    ### Step 1
    
    Create a new folder in `llama_datasets/` of the `llama_hub` Github repository. For example, in this case we would create a new folder `llama_datasets/paul_graham_essay`.
    
    In that folder, two files are required:
    - `card.json`
    - `README.md`
    
    In particular, on your local machine:
    
    ```
    cd llama_datasets/
    mkdir paul_graham_essay
    touch card.json
    touch README.md
    ```
    
    The suggestion here is to look at previously submitted `LlamaDataset`'s and modify their respective files as needed for your new dataset.
    
    In our current example, we need the `card.json` to look as follows
    
    ```json
    {
        "name": "Paul Graham Essay",
        "description": "A labelled RAG dataset based off an essay by Paul Graham, consisting of queries, reference answers, and reference contexts.",
        "numberObservations": 44,
        "containsExamplesByHumans": false,
        "containsExamplesByAI": true,
        "sourceUrls": [
            "http://www.paulgraham.com/articles.html"
        ],
        "baselines": [
            {
                "name": "llamaindex",
                "config": {
                    "chunkSize": 1024,
                    "llm": "gpt-3.5-turbo",
                    "similarityTopK": 2,
                    "embedModel": "text-embedding-ada-002"
                },
                "metrics": {
                    "contextSimilarity": 0.934,
                    "correctness": 4.239,
                    "faithfulness": 0.977,
                    "relevancy": 0.977
                },
                "codeUrl": "https://github.com/run-llama/llama_datasets/blob/main/baselines/paul_graham_essay/llamaindex_baseline.py"
            }
        ]
    }
    ```
    
    And for the `README.md`, these are pretty standard, requiring you to change the name of the dataset argument in the `download_llama_dataset()` function call.
    
    ```python
    
    # download and install dependencies for rag evaluator pack
    RagEvaluatorPack = download_llama_pack(
      "RagEvaluatorPack", "./rag_evaluator_pack"
    )
    rag_evaluator_pack = RagEvaluatorPack()
    
    # download and install dependencies for benchmark dataset
    rag_dataset, documents = download_llama_datasets(
      "PaulGrahamEssayTruncatedDataset", "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp"
    )
    
    # evaluate
    query_engine = VectorStoreIndex.as_query_engine()  # previously defined, not shown here
    rag_evaluate_pack.run(dataset=paul_graham_qa_data, query_engine=query_engine)
    ```
    
    Finally, the last item for Step 1 is to create an entry to `llama_datasets/library.json` file. In this case:
    
    ```json
        ...,
        "PaulGrahamEssayDataset": {
        "id": "llama_datasets/paul_graham_essay",
        "author": "andrei-fajardo",
        "keywords": ["rag"],
        "extra_files": ["paul_graham_essay.txt"]
      }
    ```
    
    Note: the `extra_files` field is reserved for the source files.
    
    ### Step 2 Uploading The Actual Datasets
    
    In this step, since we use Github LFS on our `llama_datasets` repo, making a contribution is exactly the same way you would make a contribution with any of our other open Github repos. That is, submit a pull request.
    
    Make a fork of the `llama_datasets` repo, and create a new folder in the `llama_datasets/` directory that matches the `id` field of the entry made in the `library.json` file. So, for this example, we'll create a new folder `llama_datasets/paul_graham_essay/`. It is here where we will add the documents and make the pull request.
    
    To this folder, add `rag_dataset.json` (it must be called this), as well as the rest of the source documents, which in our case is the `paul_graham_essay.txt` file.
    
    ```sh
    llama_datasets/paul_graham_essay/
    ├── paul_graham_essay.txt
    └── rag_dataset.json
    ```
    
    Now, simply `git add`, `git commit` and `git push` your branch, and make your PR.
    """
    logger.info("## Submitting The Pull-Requests")

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
