import spacy
from jet.logger import logger

model = "urchade/gliner_small-v2.1"
# model = "urchade/gliner_medium-v2.1"
style = "ent"

text = """
Role Overview:

We are seeking a skilled app developer to build our mobile app from scratch, integrating travel booking, relocation services, and community features. You will lead the design, development, and launch of our travel app.

Responsibilities:

Develop and maintain a scalable mobile app for iOS & Android

Integrate booking systems, payment gateways, and user profiles

Ensure seamless user experience & mobile responsiveness

Work with the team to test & refine the app before launch

Implement security features to protect user data

Qualifications:

3+ years of mobile app development (React Native, Flutter, Swift, or Kotlin)

Experience with APIs, databases, and cloud-based deployment

Strong UI/UX skills to create a user-friendly interface

Previous work on travel, booking, or e-commerce apps (preferred)

Ability to work independently & meet deadlines
"""

labels = ["role", "application", "technology stack", "qualifications"]


def determine_chunk_size(text: str) -> int:
    """Dynamically set chunk size based on text length."""
    length = len(text)

    if length < 1000:
        return 250  # Small text, use smaller chunks
    elif length < 3000:
        return 350  # Medium text, moderate chunks
    else:
        return 500  # Large text, larger chunks


# Determine best chunk size based on text length
chunk_size = determine_chunk_size(text)
logger.info(f"Dynamic chunk size set to: {chunk_size}")

# Configure SpaCy with dynamic chunk size
custom_spacy_config = {
    "gliner_model": model,
    "chunk_size": chunk_size,
    "labels": labels,
    "style": style,
}

nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)


doc = nlp(text)

logger.newline()
logger.debug("Extracted Entities:")
for entity in doc.ents:
    logger.newline()
    logger.log("Text:", entity.text, colors=["WHITE", "INFO"])
    logger.log("Label:", entity.label_, colors=["WHITE", "SUCCESS"])
    logger.log("Score:", f"{entity._.score:.4f}", colors=["WHITE", "SUCCESS"])
    logger.log("---")
