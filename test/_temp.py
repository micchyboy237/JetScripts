from typing import List, Tuple
from bertopic import BERTopic
import re

def preprocess_documents(docs: List[str], mode: str = "clean") -> List[str]:
    """Preprocess documents based on mode: clean, raw, or structured."""
    if mode == "clean":
        return [re.sub(r'http\S+|#[^\s]+|:[^\s]+:|[^\w\s]', '', doc.lower()) for doc in docs]
    elif mode == "structured":
        return [f"Summary: {doc[:150]}..." for doc in docs]  # Mimic sectioned text
    return docs  # Raw mode keeps noise

def build_hierarchy(
    docs: List[str],
    mode: str = "clean",
    threshold: float = 0.2,
    min_similarity: float = 0.3
) -> Tuple[List[int], List[int]]:
    """Builds topic hierarchy and returns topic_ids, parent_ids."""
    model = BERTopic(embedding_model="all-MiniLM-L6-v2", verbose=True)
    topics, _ = model.fit_transform(docs)
    hierarchy = model.hierarchical_topics(docs, threshold=threshold, min_similarity=min_similarity)
    return hierarchy.topic_ids, hierarchy.parent_ids

# Expanded sample data: 100 documents per format
short_docs = [
    "Great camera quality!", "Battery dies too fast #fail", "AI zoom is amazing", "Screen is vibrant",
    "Phone overheats often", "Love the fast charger", "Camera struggles in low light", "Battery life is decent",
    "UI is super smooth", "Hate the bloatware"  # Base examples
] * 10  # Repeat to reach ~100 (10x repetition for simplicity)

long_docs = [
    "The camera quality is exceptional with vibrant colors and sharp details. However, battery life is a concern, draining quickly during heavy use.",
    "AI zoom feature is innovative but struggles in low light conditions, making photos grainy. Overall, a solid phone.",
    "Battery performance needs improvement, but the sleek design and fast processor make it a great choice.",
    "Screen resolution is stunning, but the phone overheats during gaming sessions, which is frustrating.",
    "The UI is intuitive and smooth, though pre-installed apps slow it down. Camera is a highlight for photography enthusiasts.",
    "Charging speed is impressive, but battery capacity could be better for all-day use.",
    "Low-light photography is disappointing compared to competitors, but daytime shots are excellent.",
    "Build quality feels premium, but the phone lacks water resistance, which is a drawback.",
    "Processor handles multitasking well, but software updates are slow to roll out.",
    "Camera app is feature-rich, but navigating settings can be confusing for new users."  # Base examples
] * 10  # Repeat to reach ~100

noisy_docs = [
    "Camera is 🔥 #awesome", "Battery SUCKS!! #fail", "AI zoom is 😎 #tech", "Screen is 😍 #vibrant",
    "Phone gets HOT 🔥 #ugh", "Charger is ⚡ #fast", "Camera sucks in dark 😞 #fail",
    "Battery is meh 😐 #okay", "UI is slick 😎 #smooth", "Too much bloatware 😡 #annoying"
] * 10  # Repeat to reach ~100

# Process each format
clean_docs = preprocess_documents(short_docs, mode="clean")
structured_docs = preprocess_documents(long_docs, mode="structured")
raw_docs = preprocess_documents(noisy_docs, mode="raw")

# Build hierarchies
clean_topic_ids, clean_parent_ids = build_hierarchy(clean_docs, mode="clean")
structured_topic_ids, structured_parent_ids = build_hierarchy(structured_docs, mode="structured")
raw_topic_ids, raw_parent_ids = build_hierarchy(raw_docs, mode="raw")

# Example output
print(f"Clean Hierarchy (first 5): {list(zip(clean_topic_ids[:5], clean_parent_ids[:5]))}")
print(f"Structured Hierarchy (first 5): {list(zip(structured_topic_ids[:5], structured_parent_ids[:5]))}")
print(f"Raw Hierarchy (first 5): {list(zip(raw_topic_ids[:5], raw_parent_ids[:5]))}")