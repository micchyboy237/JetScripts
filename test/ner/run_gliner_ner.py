from gliner import GLiNER
from jet.logger import logger

# Initialize GLiNER with the base model
# model = "urchade/gliner_small-v2.1"
model = "urchade/gliner_medium-v2.1"

model = GLiNER.from_pretrained(model)

# Sample text for entity prediction
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

# Labels for entity prediction
labels = ["role", "app nature", "coding libraries", "qualifications"]

# Perform entity prediction
entities = model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
# for entity in entities:
#     print(entity["text"], "=>", entity["label"])

logger.newline()
logger.debug("Extracted Entities:")
for entity in entities:
    logger.newline()
    logger.log("Text:", entity['text'], colors=["WHITE", "INFO"])
    logger.log("Label:", entity['label'], colors=["WHITE", "SUCCESS"])
    logger.log("Score:", f"{entity['score']:.4f}", colors=[
               "WHITE", "SUCCESS"])
    # logger.log("Start:", f"{entity['char_start_index']}", colors=[
    #            "WHITE", "SUCCESS"])
    # logger.log("End:", f"{entity['char_end_index']}",
    #            colors=["WHITE", "SUCCESS"])
    logger.log("---")
