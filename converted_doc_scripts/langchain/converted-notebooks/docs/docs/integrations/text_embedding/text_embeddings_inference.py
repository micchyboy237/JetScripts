from jet.logger import logger
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
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
# Text Embeddings Inference

>[Hugging Face Text Embeddings Inference (TEI)](https://huggingface.co/docs/text-embeddings-inference/index) is a toolkit for deploying and serving open-source
> text embeddings and sequence classification models. `TEI` enables high-performance extraction for the most popular models,
>including `FlagEmbedding`, `Ember`, `GTE` and `E5`.

To use it within langchain, first install `huggingface-hub`.
"""
logger.info("# Text Embeddings Inference")

# %pip install --upgrade huggingface-hub

"""
Then expose an embedding model using TEI. For instance, using Docker, you can serve `BAAI/bge-large-en-v1.5` as follows:

```bash
model=BAAI/bge-large-en-v1.5
revision=refs/pr/5
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:0.6 --model-id $model --revision $revision
```

Specifics on Docker usage might vary with the underlying hardware. For example, to serve the model on Intel Gaudi/Gaudi2 hardware, refer to the [tei-gaudi repository](https://github.com/huggingface/tei-gaudi) for the relevant docker run command.

Finally, instantiate the client and embed your texts.
"""
logger.info("Then expose an embedding model using TEI. For instance, using Docker, you can serve `BAAI/bge-large-en-v1.5` as follows:")


embeddings = HuggingFaceEndpointEmbeddings(model="http://localhost:8080")

text = "What is deep learning?"

query_result = embeddings.embed_query(text)
query_result[:3]

doc_result = embeddings.embed_documents([text])

doc_result[0][:3]

logger.info("\n\n[DONE]", bright=True)