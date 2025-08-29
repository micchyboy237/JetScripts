from jet.logger import CustomLogger
from llama_index.embeddings.bedrock import BedrockEmbedding
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/bedrock.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Bedrock Embeddings
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Bedrock Embeddings")

# %pip install llama-index-embeddings-bedrock



embed_model = BedrockEmbedding(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name="<aws-region>",
    profile_name="<aws-profile>",
)

embedding = embed_model.get_text_embedding("hello world")

"""
## List supported models

To check list of supported models of Amazon Bedrock on LlamaIndex, call `BedrockEmbedding.list_supported_models()` as follows.
"""
logger.info("## List supported models")


supported_models = BedrockEmbedding.list_supported_models()
logger.debug(json.dumps(supported_models, indent=2))

"""
## Provider: Amazon
Amazon Bedrock Titan embeddings.
"""
logger.info("## Provider: Amazon")


model = BedrockEmbedding(model_name="amazon.titan-embed-g1-text-02")
embeddings = model.get_text_embedding("hello world")
logger.debug(embeddings)

"""
## Provider: Cohere

### cohere.embed-english-v3
"""
logger.info("## Provider: Cohere")

model = BedrockEmbedding(model_name="cohere.embed-english-v3")
coherePayload = ["This is a test document", "This is another test document"]

embed1 = model.get_text_embedding("This is a test document")
logger.debug(embed1)

embeddings = model.get_text_embedding_batch(coherePayload)
logger.debug(embeddings)

"""
### MultiLingual Embeddings from Cohere
"""
logger.info("### MultiLingual Embeddings from Cohere")

model = BedrockEmbedding(model_name="cohere.embed-multilingual-v3")
coherePayload = [
    "This is a test document",
    "తెలుగు అనేది ద్రావిడ భాషల కుటుంబానికి చెందిన భాష.",
    "Esto es una prueba de documento multilingüe.",
    "攻殻機動隊",
    "Combien de temps ça va prendre ?",
    "Документ проверен",
]
embeddings = model.get_text_embedding_batch(coherePayload)
logger.debug(embeddings)

logger.info("\n\n[DONE]", bright=True)