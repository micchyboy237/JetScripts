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
text = "Looking for someone to be a part of our agency. Our agency, Vean Global, is a cutting-edge web design and marketing agency specializing in Shopify development, 3D web experiences, and high-performance e-commerce solutions. We\u2019re looking for an experienced Shopify Theme Developer to help us build a fully customizable Shopify theme.\n\n\n\nWhat You'll Be Doing:\n\n\n\n- Developing a high-performance, fully customizable Shopify theme from scratch\n\n- Writing clean, maintainable, and scalable Liquid, JavaScript (Vanilla/React), HTML, and CSS code\n\n- Ensuring theme customization options are user-friendly and intuitive\n\n- Optimizing theme performance for fast loading speeds and smooth UX\n\n- Implementing dynamic content, animations, and advanced customization features\n\n- Troubleshooting and resolving theme-related issues\n\n- Working closely with our team to ensure branding, UI/UX, and functionality align with our goals\n\n\n\nWhat We\u2019re Looking For:\n\n\n\n-  Strong proficiency in Shopify's Liquid templating language\n\n- Expertise in HTML, CSS, JavaScript, and Shopify APIs\n\n- Experience with Shopify metafields and theme customizations\n\n- Strong knowledge of performance optimization best practices\n\n- Experience building custom Shopify themes (portfolio required)\n\n- Ability to work independently and meet deadlines\n\n\n\nLooking forward to working with you!"

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
