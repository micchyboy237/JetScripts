from jet.wordnet.keywords import rake_extract_keywords

# Example usage
if __name__ == "__main__":
    # Install dependencies: pip install nltk rake_nltk
    # Run nltk.download('punkt') if needed
    texts = [
        "How to bake sourdough bread.",
        "Learn advanced JavaScript and React.",
        "Python tips and tricks for developers.",
        "Understanding the basics of machine learning.",
        "Tips for cooking with cast iron skillets."
    ]
    topic = "machine learning"
    max_keywords = 3

    for i, text in enumerate(texts, 1):
        try:
            keywords = rake_extract_keywords(text, max_keywords=max_keywords)
            print(f"Text {i}: {text}")
            print(f"Keywords: {', '.join(keywords)}")
            # Optionally check if topic keywords appear
            topic_keywords = rake_extract_keywords(topic, max_keywords=2)
            relevance = any(kw in keywords for kw in topic_keywords)
            print(f"Relevant to '{topic}': {relevance}\n")
        except ImportError as e:
            print(f"Error processing text {i}: {e}")
