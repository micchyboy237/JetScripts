
from infinity_client.models import OpenAIEmbeddingInputText, OpenAIEmbeddingResult
from infinity_client.api.default import classify, embeddings, embeddings_image, rerank
from infinity_client.types import Response
from infinity_client import Cn_lient
from transformers.modeling_utils

from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# i_client = Client(base_url="https://infinity.modal.michaelfeil.eu")
i_client = Client(base_url="http://shawn-pc.local:3000")

with i_client as client:
    embeds: OpenAIEmbeddingResult = embeddings.sync(client=client, body=OpenAIEmbeddingInputText.from_dict({
        "input": ["Hello, this is a sentence-to-embed", "Hello, my cat is cute"],
        "model": "michaelfeil/bge-small-en-v1.5",
    }))
    save_file(embeds, f"{OUTPUT_DIR}/embeds.json")
    