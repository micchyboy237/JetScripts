from jet.logger import logger
from langchain_community.document_loaders import TensorflowDatasetLoader
import os
import shutil


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
# TensorFlow Datasets

>[TensorFlow Datasets](https://www.tensorflow.org/datasets) is a collection of datasets ready to use,
> with TensorFlow or other Python ML frameworks, such as Jax. All datasets are exposed
> as [tf.data.Datasets](https://www.tensorflow.org/api_docs/python/tf/data/Dataset),
> enabling easy-to-use and high-performance input pipelines. To get started see
> the [guide](https://www.tensorflow.org/datasets/overview) and
> the [list of datasets](https://www.tensorflow.org/datasets/catalog/overview#all_datasets).



## Installation and Setup

You need to install `tensorflow` and `tensorflow-datasets` python packages.
"""
logger.info("# TensorFlow Datasets")

pip install tensorflow

"""

"""

pip install tensorflow-dataset

"""
## Document Loader

See a [usage example](/docs/integrations/document_loaders/tensorflow_datasets).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)