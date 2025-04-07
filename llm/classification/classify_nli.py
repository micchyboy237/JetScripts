from transformers import pipeline

# Load a pretrained NLI pipeline
nli = pipeline("text-classification", model="facebook/bart-large-mnli")

# Real-world example inputs
examples = [
    {
        "premise": "A customer is waiting in line at the coffee shop.",
        "hypothesis": "Someone is buying a drink.",
    },
    {
        "premise": "The server placed the bill on the table.",
        "hypothesis": "The food was free.",
    },
    {
        "premise": "She’s walking her dog in the park.",
        "hypothesis": "She is exercising.",
    },
]

# Inference
for i, ex in enumerate(examples, 1):
    result = nli(f"{ex['premise']} </s> {ex['hypothesis']}")
    print(f"Example {i}:")
    print(f"  Premise:    {ex['premise']}")
    print(f"  Hypothesis: {ex['hypothesis']}")
    print(
        f"  Prediction: {result[0]['label']} (score: {result[0]['score']:.4f})\n")
