from jet.logger import logger
from langchain_community.document_loaders import TensorflowDatasetLoader
from langchain_core.documents import Document
import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds


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

>[TensorFlow Datasets](https://www.tensorflow.org/datasets) is a collection of datasets ready to use, with TensorFlow or other Python ML frameworks, such as Jax. All datasets are exposed as [tf.data.Datasets](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), enabling easy-to-use and high-performance input pipelines. To get started see the [guide](https://www.tensorflow.org/datasets/overview) and the [list of datasets](https://www.tensorflow.org/datasets/catalog/overview#all_datasets).

This notebook shows how to load `TensorFlow Datasets` into a Document format that we can use downstream.

## Installation

You need to install `tensorflow` and `tensorflow-datasets` python packages.
"""
logger.info("# TensorFlow Datasets")

# %pip install --upgrade --quiet  tensorflow

# %pip install --upgrade --quiet  tensorflow-datasets

"""
## Example

As an example, we use the [`mlqa/en` dataset](https://www.tensorflow.org/datasets/catalog/mlqa#mlqaen).

>`MLQA` (`Multilingual Question Answering Dataset`) is a benchmark dataset for evaluating multilingual question answering performance. The dataset consists of 7 languages: Arabic, German, Spanish, English, Hindi, Vietnamese, Chinese.
>
>- Homepage: https://github.com/facebookresearch/MLQA
>- Source code: `tfds.datasets.mlqa.Builder`
>- Download size: 72.21 MiB
"""
logger.info("## Example")

FeaturesDict(
    {
        "answers": Sequence(
            {
                "answer_start": int32,
                "text": Text(shape=(), dtype=string),
            }
        ),
        "context": Text(shape=(), dtype=string),
        "id": string,
        "question": Text(shape=(), dtype=string),
        "title": Text(shape=(), dtype=string),
    }
)


ds = tfds.load("mlqa/en", split="test")
ds = ds.take(1)  # Only take a single example
ds

"""
Now we have to create a custom function to convert dataset sample into a Document.

This is a requirement. There is no standard format for the TF datasets that's why we need to make a custom transformation function.

Let's use `context` field as the `Document.page_content` and place other fields in the `Document.metadata`.
"""
logger.info("Now we have to create a custom function to convert dataset sample into a Document.")



def decode_to_str(item: tf.Tensor) -> str:
    return item.numpy().decode("utf-8")


def mlqaen_example_to_document(example: dict) -> Document:
    return Document(
        page_content=decode_to_str(example["context"]),
        metadata={
            "id": decode_to_str(example["id"]),
            "title": decode_to_str(example["title"]),
            "question": decode_to_str(example["question"]),
            "answer": decode_to_str(example["answers"]["text"][0]),
        },
    )


for example in ds:
    doc = mlqaen_example_to_document(example)
    logger.debug(doc)
    break


loader = TensorflowDatasetLoader(
    dataset_name="mlqa/en",
    split_name="test",
    load_max_docs=3,
    sample_to_document_function=mlqaen_example_to_document,
)

"""
`TensorflowDatasetLoader` has these parameters:
- `dataset_name`: the name of the dataset to load
- `split_name`: the name of the split to load. Defaults to "train".
- `load_max_docs`: a limit to the number of loaded documents. Defaults to 100.
- `sample_to_document_function`: a function that converts a dataset sample to a Document
"""

docs = loader.load()
len(docs)

docs[0].page_content

docs[0].metadata

logger.info("\n\n[DONE]", bright=True)