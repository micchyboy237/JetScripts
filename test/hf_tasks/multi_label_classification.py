from transformers import pipeline

# Load the multi-label classification pipeline
classifier = pipeline("text-classification",
                      model="distilbert-base-uncased", return_all_scores=True)

# Example text
text = "The new phone has amazing battery life and performance."

# Run classification
result = classifier(text)
print(result)
