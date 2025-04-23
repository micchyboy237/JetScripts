from transformers import pipeline

# Load the zero-shot classification pipeline using BART
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

# Example text
text = "The new phone has amazing battery life and performance."

# Run classification (note: we're not defining candidate labels)
result = classifier(text)
print(result)
