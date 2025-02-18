import spacy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import re

# Step 1: Preprocess the text (tokenization, lemmatization, and remove stopwords)


def preprocess_text(text: str):
    # Remove non-alphabetic characters and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase the text
    text = text.lower()
    return text

# Step 2: Extract meaningful keywords/phrases (noun phrases or other significant terms)


def extract_keywords(text: str, model, tokenizer):
    # Use spaCy for part-of-speech tagging and extracting noun phrases
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Extract noun phrases (or any other custom keyword extraction logic)
    keywords = [chunk.text for chunk in doc.noun_chunks if len(
        chunk.text.split()) > 1]

    # Get embeddings for these keywords/phrases
    inputs = tokenizer(keywords, return_tensors='pt',
                       padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(
            **inputs).last_hidden_state.mean(dim=1).cpu().numpy()

    return keywords, embeddings

# Step 3: Perform K-means clustering to group similar keywords


def cluster_keywords(keywords: list, embeddings: np.ndarray, n_clusters: int = 5):
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(similarity_matrix)

    return kmeans.labels_

# Step 4: Main pipeline to process the long text


def group_similar_keywords(text: str, n_clusters: int = 5):
    # Initialize the pre-trained CodeBERT model
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaModel.from_pretrained('microsoft/codebert-base')

    # Preprocess text
    processed_text = preprocess_text(text)

    # Extract keywords and their embeddings
    keywords, embeddings = extract_keywords(processed_text, model, tokenizer)

    # Cluster the keywords
    clusters = cluster_keywords(keywords, embeddings, n_clusters)

    return keywords, clusters


# Example Usage:
if __name__ == "__main__":
    text = """Looking for someone to be a part of our agency. Our agency, Vean Global, is a cutting-edge web design and marketing agency specializing in Shopify development, 3D web experiences, and high-performance e-commerce solutions. We’re looking for an experienced Shopify Theme Developer to help us build a fully customizable Shopify theme.

    What You'll Be Doing:
    - Developing a high-performance, fully customizable Shopify theme from scratch
    - Writing clean, maintainable, and scalable Liquid, JavaScript (Vanilla/React), HTML, and CSS code
    - Ensuring theme customization options are user-friendly and intuitive
    - Optimizing theme performance for fast loading speeds and smooth UX
    - Implementing dynamic content, animations, and advanced customization features
    - Troubleshooting and resolving theme-related issues
    - Working closely with our team to ensure branding, UI/UX, and functionality align with our goals

    What We’re Looking For:
    - Strong proficiency in Shopify's Liquid templating language
    - Expertise in HTML, CSS, JavaScript, and Shopify APIs
    - Experience with Shopify metafields and theme customizations
    - Strong knowledge of performance optimization best practices
    - Experience building custom Shopify themes (portfolio required)
    - Ability to work independently and meet deadlines"""

    # Group keywords into clusters
    keywords, clusters = group_similar_keywords(text, n_clusters=3)

    # Print grouped keywords
    print("Grouped Keywords by Clusters:")
    for i in range(len(set(clusters))):
        cluster_keywords = [keywords[idx]
                            for idx, label in enumerate(clusters) if label == i]
        print(f"Cluster {i+1}: {', '.join(cluster_keywords)}")
