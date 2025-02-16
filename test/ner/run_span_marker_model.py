import transformers
import os
from span_marker import SpanMarkerModel
from span_marker.tokenizer import SpanMarkerTokenizer
from jet.logger import logger
os.environ["TOKENIZERS_PARALLELISM"] = "true"


print("transformers:", transformers.__version__)

# Download from the 🤗 Hub
model = SpanMarkerModel.from_pretrained(
    "tomaarsen/span-marker-mbert-base-multinerd"
    # "tomaarsen/span-marker-roberta-large-ontonotes5"
)
tokenizer: SpanMarkerTokenizer = model.tokenizer
# Resolves error in data collator trying to find missing pad_token_id
tokenizer.pad_token_id = tokenizer.pad_token_type_id

# Input text
text = """Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the 
Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her 
death in 30 BCE."""

logger.info(text)
logger.debug("Predicting...")
# Predict entities
entities = model.predict(text)

logger.debug("Extracted Entities:")
for entity in entities:
    logger.newline()
    logger.log("Text:", entity['span'], colors=["WHITE", "INFO"])
    logger.log("Label:", entity['label'], colors=["WHITE", "SUCCESS"])
    logger.log("Confidence:", f"{entity['score']:.4f}", colors=[
               "WHITE", "SUCCESS"])
    logger.log("Start:", f"{entity['char_start_index']}", colors=[
               "WHITE", "SUCCESS"])
    logger.log("End:", f"{entity['char_end_index']}",
               colors=["WHITE", "SUCCESS"])
    logger.log("---")
