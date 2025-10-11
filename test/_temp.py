from typing import List, Dict, Any, Union
from pathlib import Path
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.utils.url_utils import clean_links
from tqdm import tqdm
from jet.adapters.bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import pandas as pd

from jet.file.utils import load_file, save_file
from jet.logger import logger
import os
import shutil

from jet.wordnet.text_chunker import chunk_texts

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

OUTPUT_DIR = Path(OUTPUT_DIR)

EMBED_MODEL = "embeddinggemma"

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)


def load_sample_data():
    """Load sample dataset from local for topic modeling."""
    logger.info("Loading sample dataset...")
    headers_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/all_headers.json"
    headers_dict = load_file(headers_file)
    headers: List[HeaderDoc] = [h for h_list in headers_dict.values() for h in h_list]
    documents = [f"{doc["header"]}\n\n{doc['content']}" for doc in headers]

    # Clean all links
    documents = [clean_markdown_links(doc) for doc in documents]
    documents = [clean_links(doc) for doc in documents]

    documents = chunk_texts(
        documents,
        chunk_size=64,
        chunk_overlap=32,
        model=EMBED_MODEL,
    )
    timestamps = [i % 10 for i in range(len(documents))]  # Synthetic timestamps
    save_file(documents, f"{OUTPUT_DIR}/documents.json")
    save_file(timestamps, f"{OUTPUT_DIR}/timestamps.json")
    return documents, timestamps

def save_topic_info(
    topic_model: BERTopic,
    output_path: Path,
    data: Union[pd.DataFrame, List[Dict[str, Any]]] = None,
    top_n_words: int = 10,
    is_hierarchical: bool = False
) -> None:
    """Save topic information to a JSON file with enhanced metadata."""
    from jet.file.utils import save_file
    if data is None:
        data = topic_model.get_topic_info()
    
    if is_hierarchical:
        formatted_data = []
        for row in data.to_dict(orient="records"):
            topic_data = {
                "parent_id": row["Parent_ID"],
                "parent_name": row["Parent_Name"],
                "topics": [
                    {
                        "topic_id": topic_id,
                        "top_words": [{"word": word[0], "weight": float(word[1])} for word in topic_model.get_topic(topic_id)[:top_n_words]],
                        "representative_docs": topic_model.get_representative_docs(topic_id)[:3]
                    }
                    for topic_id in row["Topics"]
                ],
                "child_left_id": row["Child_Left_ID"],
                "child_left_name": row["Child_Left_Name"],
                "child_right_id": row["Child_Right_ID"],
                "child_right_name": row["Child_Right_Name"],
                "distance": float(row["Distance"])
            }
            formatted_data.append(topic_data)
    else:
        formatted_data = [
            {
                "topic_id": row["Topic"],
                "count": row["Count"],
                "name": row["Name"],
                "top_words": [{"word": word[0], "weight": float(word[1])} for word in topic_model.get_topic(row["Topic"])[:top_n_words]],
                "representative_docs": topic_model.get_representative_docs(row["Topic"])[:3]
            }
            for row in data.to_dict(orient="records") if row["Topic"] != -1
        ]
    
    save_file(formatted_data, output_path, verbose=True)

def get_vectorizer(total_docs: int) -> CountVectorizer:
    """Create a CountVectorizer with stopword removal and dynamic df thresholds."""
    stop_words = list(set(stopwords.words('english')))
    min_df = 2
    max_df = min(0.95, max(0.5, (total_docs - 1) / total_docs))  # Ensure max_df is valid
    logger.info(f"Configuring vectorizer with min_df={min_df}, max_df={max_df}")
    return CountVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df)

def example_base_topic_modeling():
    """Demonstrate basic topic modeling with BERTopic."""
    logger.info("Starting basic topic modeling example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic(vectorizer_model=get_vectorizer(len(documents)), verbose=True)
    logger.info("Fitting topic model...")
    with tqdm(total=len(documents), desc="Processing documents") as pbar:
        topics, _ = topic_model.fit_transform(documents)
        pbar.update(len(documents))
    
    # Save topic information
    output_path = OUTPUT_DIR / "base_topic_modeling.json"
    save_topic_info(topic_model, output_path)
    
    # Log topic information
    topic_info = topic_model.get_topic_info()
    logger.info(f"Found {len(topic_info)} topics")
    for topic in topic_info['Topic']:
        if topic != -1:
            words = topic_model.get_topic(topic)[:5]
            logger.info(f"Topic {topic}: {', '.join([word[0] for word in words])}")
    
    # Save and load model
    logger.info("Saving model...")
    topic_model.save("model_dir", serialization="pytorch", save_ctfidf=True)
    logger.info("Loading model...")
    loaded_model = BERTopic.load("model_dir")
    logger.info("Model loaded successfully")
    
    return topic_model

def example_topic_prediction():
    """Demonstrate predicting topics for new documents."""
    logger.info("Starting topic prediction example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic(vectorizer_model=get_vectorizer(len(documents)), verbose=True)
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Predict topics for new documents
    new_docs = [
        "The new smartphone features advanced AI capabilities.",
        "Global warming is affecting climate patterns worldwide."
    ]
    logger.info("Predicting topics for new documents...")
    with tqdm(total=len(new_docs), desc="Predicting topics") as pbar:
        topics, probs = topic_model.transform(new_docs)
        pbar.update(len(new_docs))
    
    # Save prediction results
    output_path = OUTPUT_DIR / "topic_predictions.jsonl"
    predictions = [
        {
            "document": doc,
            "topic_id": topic,
            "probability": float(prob),
            "top_words": [word[0] for word in topic_model.get_topic(topic)[:5]]
        }
        for doc, topic, prob in zip(new_docs, topics, probs)
    ]
    save_file(predictions, output_path, verbose=True, append=False)
    
    # Log predictions
    for i, (topic, prob) in enumerate(zip(topics, probs)):
        words = topic_model.get_topic(topic)[:5]
        logger.info(f"Document {i+1} assigned to Topic {topic} (Probability: {prob:.2f}): "
                   f"{', '.join([word[0] for word in words])}")
    
    return topics, probs

def example_topics_over_time():
    """Demonstrate analyzing topics over time."""
    logger.info("Starting topics over time example...")
    documents, timestamps = load_sample_data()
    topic_model = BERTopic(vectorizer_model=get_vectorizer(len(documents)), verbose=True)
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Analyze topics over time
    logger.info("Computing topics over time...")
    with tqdm(total=len(timestamps), desc="Analyzing topics over time") as pbar:
        topics_over_time = topic_model.topics_over_time(documents, timestamps)
        pbar.update(len(timestamps))
    
    # Save topics over time
    output_path = OUTPUT_DIR / "topics_over_time.json"
    formatted_data = topics_over_time.to_dict(orient="records")
    for row in formatted_data:
        row["top_words"] = [word[0] for word in topic_model.get_topic(row["Topic"])[:5]]
    save_file(formatted_data, output_path, verbose=True)
    
    # Log results
    logger.info("Topics over time analysis complete")
    logger.info(f"Total frequency: {topics_over_time.Frequency.sum()}")
    logger.info(f"Unique topics: {len(topics_over_time.Topic.unique())}")
    
    return topics_over_time

def example_hierarchical_topics():
    """Demonstrate hierarchical topic modeling."""
    logger.info("Starting hierarchical topics example...")
    documents, _ = load_sample_data()
    if not documents:
        logger.error("No documents available for topic modeling")
        return None
    topic_model = BERTopic(vectorizer_model=get_vectorizer(len(documents)), min_topic_size=10, verbose=True)
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logger.info("Computing hierarchical topics...")
    with tqdm(total=len(documents), desc="Building hierarchy") as pbar:
        hier_topics = topic_model.hierarchical_topics(documents)
        pbar.update(len(documents))
    
    output_path = OUTPUT_DIR / "hierarchical_topics.json"
    save_topic_info(topic_model, output_path, data=hier_topics, is_hierarchical=True)
    
    output_tree_path = OUTPUT_DIR / "hierarchical_topic_tree.txt"
    tree = topic_model.get_topic_tree(hier_topics, tight_layout=False)
    from jet.file.utils import save_file
    save_file(tree, output_tree_path, verbose=True)
    
    logger.info("Hierarchical topic tree:\n" + tree)
    
    return hier_topics

def example_topic_search():
    """Demonstrate searching for topics similar to a query."""
    logger.info("Starting topic search example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic(vectorizer_model=get_vectorizer(len(documents)), verbose=True)
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Search for similar topics
    query = "artificial intelligence"
    logger.info(f"Searching for topics similar to: {query}")
    similar_topics, similarity = topic_model.find_topics(query, top_n=2)
    
    # Save search results
    output_path = OUTPUT_DIR / "topic_search.json"
    search_results = [
        {
            "query": query,
            "topic_id": topic,
            "similarity": float(sim),
            "top_words": [word[0] for word in topic_model.get_topic(topic)[:5]]
        }
        for topic, sim in zip(similar_topics, similarity)
    ]
    save_file(search_results, output_path, verbose=True)
    
    # Log results
    for topic, sim in zip(similar_topics, similarity):
        words = topic_model.get_topic(topic)[:5]
        logger.info(f"Topic {topic} (Similarity: {sim:.2f}): "
                   f"{', '.join([word[0] for word in words])}")
    
    return similar_topics, similarity

def example_topic_reduction_and_update():
    """Demonstrate reducing topics and updating topic representations."""
    logger.info("Starting topic reduction and update example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic(vectorizer_model=get_vectorizer(len(documents)), verbose=True)
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Reduce topics
    original_topics = len(topic_model.get_topic_info())
    nr_topics = max(2, original_topics - 1)
    logger.info(f"Reducing topics from {original_topics} to {nr_topics}...")
    with tqdm(total=len(documents), desc="Reducing topics") as pbar:
        topic_model.reduce_topics(documents, nr_topics=nr_topics)
        pbar.update(len(documents))
    
    # Save reduced topics
    output_path = OUTPUT_DIR / "reduced_topics.json"
    save_topic_info(topic_model, output_path)
    
    # Update topic representations
    logger.info("Updating topic representations with bigrams...")
    original_vectorizer = topic_model.vectorizer_model
    topic_model.update_topics(documents, n_gram_range=(2, 2))
    
    # Save bigram topics
    output_bigrams_path = OUTPUT_DIR / "bigram_topics.json"
    save_topic_info(topic_model, output_bigrams_path)
    
    # Restore original vectorizer
    logger.info("Restoring original vectorizer...")
    topic_model.update_topics(documents, vectorizer_model=original_vectorizer)
    
    # Log results
    logger.info(f"Topics reduced to {len(topic_model.get_topic_info())}")
    
    return topic_model

def example_topic_merging():
    """Demonstrate merging topics."""
    logger.info("Starting topic merging example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic(vectorizer_model=get_vectorizer(len(documents)), verbose=True)
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Merge topics
    topics_to_merge = [0, 1]
    logger.info(f"Merging topics {topics_to_merge}...")
    with tqdm(total=len(documents), desc="Merging topics") as pbar:
        topic_model.merge_topics(documents, topics_to_merge)
        pbar.update(len(documents))
    
    # Save merged topics
    output_path = OUTPUT_DIR / "merged_topics.json"
    save_topic_info(topic_model, output_path)
    
    # Log results
    logger.info(f"Topics after merging: {len(topic_model.get_topic_info())}")
    
    return topic_model

if __name__ == "__main__":


    logger.info("Running BERTopic usage examples...")
    example_hierarchical_topics()
    example_topic_reduction_and_update()
    example_base_topic_modeling()
    example_topic_prediction()
    example_topics_over_time()
    example_topic_search()
    example_topic_merging()
    logger.info("All examples completed successfully.")
