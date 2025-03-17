#  Setup Index

import logging
from tqdm import tqdm
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting Indexing Process...")

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(nbits=2, root="data/experiments")

        logger.info("Initializing Indexer with checkpoint: data/checkpoint")
        indexer = Indexer(checkpoint="data/checkpoint", config=config)

        collection_path = "data/MSMARCO/collection.tsv"
        logger.info(f"Indexing documents from {collection_path}")

        with open(collection_path, "r", encoding="utf-8") as file:
            num_lines = sum(1 for _ in file)  # Count total lines for progress bar

        with tqdm(total=num_lines, desc="Indexing Progress") as pbar:
            indexer.index(name="msmarco.nbits=2", collection=collection_path)

        logger.info("Indexing completed successfully!")


# Retrieval

import logging
from tqdm import tqdm
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting Retrieval Process...")

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(root="data/experiments")

        logger.info("Initializing Searcher for index: msmarco.nbits=2")
        searcher = Searcher(index="msmarco.nbits=2", config=config)

        queries_path = "data/MSMARCO/queries.dev.small.tsv"
        queries = Queries(queries_path)

        logger.info(f"Loaded {len(queries)} queries from {queries_path}")

        with tqdm(total=len(queries), desc="Retrieval Progress") as pbar:
            ranking = searcher.search_all(queries, k=100)

        output_path = "msmarco.nbits=2.ranking.tsv"
        ranking.save(output_path)
        logger.info(f"Retrieval results saved to {output_path}")



# from colbert import ColBERT, Searcher

# # Load the model and index
# checkpoint = "colbert-ir/colbertv2.0"
# index_path = "path_to_index"  # You need to build an index before searching
# searcher = Searcher(index=index_path, checkpoint=checkpoint)

# # Example query
# query = "What is artificial intelligence?"
# results = searcher.search(query, k=3)

# # Display results
# for doc_id, score in zip(results[0], results[1]):
#     print(f"Score: {score:.4f} - Document ID: {doc_id}")
