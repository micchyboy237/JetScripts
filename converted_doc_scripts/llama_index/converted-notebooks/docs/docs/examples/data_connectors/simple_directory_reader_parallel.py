from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pstats import SortKey
import cProfile, pstats
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Parallel Processing SimpleDirectoryReader

In this notebook, we demonstrate how to use parallel processing when loading data with `SimpleDirectoryReader`. Parallel processing can be useful with heavier workloads i.e., loading from a directory consisting of many files. (NOTE: if using Windows, you may see less gains when using parallel processing for loading data. This has to do with the differences between how multiprocess works in linux/mac and windows e.g., see [here](https://pythonforthelab.com/blog/differences-between-multiprocessing-windows-and-linux/) or [here](https://stackoverflow.com/questions/52465237/multiprocessing-slower-than-serial-processing-in-windows-but-not-in-linux))
"""
logger.info("# Parallel Processing SimpleDirectoryReader")


"""
In this demo, we'll use the `PatronusAIFinanceBenchDataset` llama-dataset from [llamahub](https://llamahub.ai). This dataset is based off of a set of 32 PDF files which are included in the download from llamahub.
"""
logger.info("In this demo, we'll use the `PatronusAIFinanceBenchDataset` llama-dataset from [llamahub](https://llamahub.ai). This dataset is based off of a set of 32 PDF files which are included in the download from llamahub.")

# !llamaindex-cli download-llamadataset PatronusAIFinanceBenchDataset --download-dir ./data


reader = SimpleDirectoryReader(input_dir="./data/source_files")

"""
### Sequential Load

Sequential loading is the default behaviour and can be executed via the `load_data()` method.
"""
logger.info("### Sequential Load")

documents = reader.load_data()
len(documents)

cProfile.run("reader.load_data()", "oldstats")
p = pstats.Stats("oldstats")
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(15)

"""
### Parallel Load

To load using parallel processes, we set `num_workers` to a positive integer value.
"""
logger.info("### Parallel Load")

documents = reader.load_data(num_workers=10)

len(documents)

cProfile.run("reader.load_data(num_workers=10)", "newstats")
p = pstats.Stats("newstats")
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(15)

"""
### In Conclusion
"""
logger.info("### In Conclusion")

391 / 30

"""
As one can observe from the results above, there is a ~13x speed up (or 1200% speed increase) when using parallel processing when loading from a directory with many files.
"""
logger.info("As one can observe from the results above, there is a ~13x speed up (or 1200% speed increase) when using parallel processing when loading from a directory with many files.")

logger.info("\n\n[DONE]", bright=True)