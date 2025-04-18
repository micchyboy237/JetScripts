from transformers import pipeline

# Initialize a fine-tuned text classifier (replace with your fine-tuned model)
classifier = pipeline("text-classification",
                      model="distilbert-base-uncased-finetuned-custom-relevance")

# Filter results
filtered_results = []
for result in data["results"]:
    text = result["text"]
    classification = classifier(text)
    if classification[0]["label"] == "RELEVANT" and classification[0]["score"] > 0.7:
        filtered_results.append(result)

# Output filtered results
for result in filtered_results:
    print(result["text"])
