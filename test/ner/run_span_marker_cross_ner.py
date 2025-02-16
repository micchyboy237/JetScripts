from span_marker import SpanMarkerModel

# Load the pre-trained model
model = SpanMarkerModel.from_pretrained(
    "tomaarsen/span-marker-bert-base-cross-ner")

# Example text
text = "Barack Obama was born in Hawaii."

# Predict entities
entities = model.predict(text)

print(entities)
