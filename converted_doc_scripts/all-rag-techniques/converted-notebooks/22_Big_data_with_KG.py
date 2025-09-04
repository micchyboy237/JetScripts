from IPython.display import HTML, display
from collections import Counter
from datasets import load_dataset
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from pyvis.network import Network
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD, SKOS # Added SKOS for altLabel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
import spacy
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## End-to-End Pipeline: Big Data with Knowledge Graph (Book-Referenced)

### Goal:
Transform news articles about technology company acquisitions into a structured Knowledge Graph, using modern techniques for extraction, refinement, and reasoning — guided by foundational principles outlined in a conceptual book.

### Dataset: CNN/DailyMail

### Approach Overview:
This notebook walks through a multi-phase process:
1.  **Data Acquisition & Preparation:** Sourcing and cleaning raw news text.
2.  **Information Extraction:** Identifying key entities (organizations, people, money, dates) and the relationships between them (e.g., 'acquire', 'invested_in').
3.  **Knowledge Graph Construction:** Structuring the extracted information into RDF triples, forming the nodes and edges of our KG.
4.  **KG Refinement (Conceptual):** Using embeddings to represent KG components and conceptually exploring link prediction.
5.  **Persistence & Utilization:** Storing, querying (SPARQL), and visualizing the KG.

We will leverage Large Language Models (LLMs) for complex NLP tasks like nuanced entity and relationship extraction, while also using traditional libraries like spaCy for initial exploration and `rdflib` for KG management.

# Table of Contents

- [End-to-End Pipeline: Big Data with Knowledge Graph (Book-Referenced)](#intro-0)
  - [Initial Setup: Imports and Configuration](#intro-setup)
    - [Initialize LLM Client and spaCy Model](#llm-spacy-init-desc)
    - [Define RDF Namespaces](#namespace-init-desc)
- [Phase 1: Data Acquisition and Preparation](#phase1)
  - [Step 1.1: Data Acquisition](#step1-1-desc)
    - [Execute Data Acquisition](#data-acquisition-exec-desc)
  - [Step 1.2: Data Cleaning & Preprocessing](#step1-2-desc)
    - [Execute Data Cleaning](#data-cleaning-exec-desc)
- [Phase 2: Information Extraction](#phase2)
  - [Step 2.1: Entity Extraction (Named Entity Recognition - NER)](#step2-1-desc)
    - [2.1.1: Entity Exploration with spaCy - Function Definition](#step2-1-1-spacy-desc)
    - [2.1.1: Entity Exploration with spaCy - Plotting Function Definition](#plot_entity_distribution_func_def_desc)
    - [2.1.1: Entity Exploration with spaCy - Execution](#spacy-ner-exec-desc)
    - [Generic LLM Call Function Definition](#llm-call-func-def-desc)
    - [2.1.2: Entity Type Selection using LLM - Execution](#step2-1-2-llm-type-selection-desc)
    - [LLM JSON Output Parsing Function Definition](#parse_llm_json_func_def_desc)
    - [2.1.3: Targeted Entity Extraction using LLM - Execution](#step2-1-3-llm-ner-exec-desc)
  - [Step 2.2: Relationship Extraction](#step2-2-desc)
- [Phase 3: Knowledge Graph Construction](#phase3)
  - [Step 3.1: Entity Disambiguation & Linking (Simplified) - Normalization Function](#step3-1-normalize-entity-text-func-def-desc)
    - [Execute Entity Normalization and URI Generation](#entity-normalization-exec-desc)
  - [Step 3.2: Schema/Ontology Alignment - RDF Type Mapping Function](#step3-2-rdf-type-func-def-desc)
    - [Schema/Ontology Alignment - RDF Predicate Mapping Function](#step3-2-rdf-predicate-func-def-desc)
    - [Schema/Ontology Alignment - Examples](#schema-alignment-example-desc)
  - [Step 3.3: Triple Generation](#step3-3-triple-generation-exec-desc)
- [Phase 4: Knowledge Graph Refinement Using Embeddings](#phase4)
  - [Step 4.1: Generate KG Embeddings - Embedding Function Definition](#step4-1-embedding-func-def-desc)
    - [Generate KG Embeddings - Execution](#kg-embedding-exec-desc)
  - [Step 4.2: Link Prediction (Knowledge Discovery - Conceptual) - Cosine Similarity Function](#step4-2-cosine-sim-func-def-desc)
    - [Link Prediction (Conceptual) - Similarity Calculation Example](#link-prediction-exec-desc)
  - [Step 4.3: Add Predicted Links (Optional & Conceptual) - Function Definition](#step4-3-add-inferred-func-def-desc)
    - [Add Predicted Links (Conceptual) - Execution Example](#add-predicted-links-exec-desc)
- [Phase 5: Persistence and Utilization](#phase5)
  - [Step 5.1: Knowledge Graph Storage - Save Function Definition](#step5-1-save-graph-func-def-desc)
    - [Knowledge Graph Storage - Execution](#kg-storage-exec-desc)
  - [Step 5.2: Querying and Analysis - SPARQL Execution Function](#step5-2-sparql-func-def-desc)
    - [SPARQL Querying and Analysis - Execution Examples](#sparql-querying-exec-desc)
  - [Step 5.3: Visualization (Optional) - Visualization Function Definition](#step5-3-viz-func-def-desc)
    - [KG Visualization - Execution](#visualization-exec-desc)
- [Conclusion and Future Work](#conclusion)

### Initial Setup: Imports and Configuration

**Theory:**
Before any data processing or analysis can begin, we need to set up our environment. This involves:
*   **Importing Libraries:** Bringing in the necessary Python packages. These include `datasets` for data loading, `openai` for interacting with LLMs, `spacy` for foundational NLP, `rdflib` for Knowledge Graph manipulation, `re` for text processing with regular expressions, `json` for handling LLM outputs, `matplotlib` and `pyvis` for visualization, and standard libraries like `os`, `collections`, and `tqdm`.
*   **API Configuration:** Setting up credentials and endpoints for external services, specifically the Nebius LLM API. **Security Note:** In a production environment, API keys should never be hardcoded. Use environment variables or secure secret management systems.
*   **Model Initialization:** Loading pre-trained models like spaCy's `en_core_web_sm` for basic NLP tasks and configuring the LLM client to use specific models deployed on Nebius for generation and embeddings.
*   **Namespace Definitions:** For RDF-based Knowledge Graphs, namespaces (like `EX` for our custom terms, `SCHEMA` for schema.org) are crucial for creating unique and resolvable URIs for entities and properties. This aligns with the Linked Data principles.
"""
logger.info("## End-to-End Pipeline: Big Data with Knowledge Graph (Book-Referenced)")







NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "your_nebius_api_key_here") # Replace with your actual API key
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"

TEXT_GEN_MODEL_NAME = "deepseek-ai/DeepSeek-V3" # e.g., phi-4, deepseek or any other model
EMBEDDING_MODEL_NAME = "BAAI/bge-multilingual-gemma2" # e.g., text-embedding-ada-002, BAAI/bge-multilingual-gemma2 or any other model

logger.debug("Libraries imported.")

"""
**Output Explanation:**
This block simply confirms that the necessary libraries have been imported without error.

#### Initialize LLM Client and spaCy Model

**Theory:**
Here, we instantiate the clients for our primary NLP tools:
*   **MLX Client:** Configured to point to the Nebius API. This client will be used to send requests to the deployed LLM for tasks like entity extraction, relation extraction, and generating embeddings. A basic check is performed to see if the configuration parameters are set.
*   **spaCy Model:** We load `en_core_web_sm`, a small English model from spaCy. This model provides efficient capabilities for tokenization, part-of-speech tagging, lemmatization, and basic Named Entity Recognition (NER). It's useful for initial text exploration and can complement LLM-based approaches.
"""
logger.info("#### Initialize LLM Client and spaCy Model")

client = None # Initialize client to None
if NEBIUS_API_KEY != "YOUR_NEBIUS_API_KEY" and NEBIUS_BASE_URL != "YOUR_NEBIUS_BASE_URL" and TEXT_GEN_MODEL_NAME != "YOUR_TEXT_GENERATION_MODEL_NAME":
    try:
        client = MLX(
            base_url=NEBIUS_BASE_URL,
            api_key=NEBIUS_API_KEY
        )
        logger.debug(f"MLX client initialized for base_url: {NEBIUS_BASE_URL} using model: {TEXT_GEN_MODEL_NAME}")
    except Exception as e:
        logger.debug(f"Error initializing MLX client: {e}")
        client = None # Ensure client is None if initialization fails
else:
    logger.debug("Warning: MLX client not fully configured. LLM features will be disabled. Please set NEBIUS_API_KEY, NEBIUS_BASE_URL, and TEXT_GEN_MODEL_NAME.")

nlp_spacy = None # Initialize nlp_spacy to None
try:
    nlp_spacy = spacy.load("en_core_web_sm")
    logger.debug("spaCy model 'en_core_web_sm' loaded.")
except OSError:
    logger.debug("spaCy model 'en_core_web_sm' not found. Downloading... (This might take a moment)")
    try:
        spacy.cli.download("en_core_web_sm")
        nlp_spacy = spacy.load("en_core_web_sm")
        logger.debug("spaCy model 'en_core_web_sm' downloaded and loaded successfully.")
    except Exception as e:
        logger.debug(f"Failed to download or load spaCy model: {e}")
        logger.debug("Please try: python -m spacy download en_core_web_sm in your terminal and restart the kernel.")
        nlp_spacy = None # Ensure nlp_spacy is None if loading fails

"""
**Output Explanation:**
This block prints messages indicating the status of the MLX client and spaCy model initialization. Warnings are shown if configurations are missing or models can't be loaded.

#### Define RDF Namespaces

**Theory:**
In RDF, namespaces are used to avoid naming conflicts and to provide context for terms (URIs). 
*   `EX`: A custom namespace for terms specific to our project (e.g., our entities and relationships if not mapped to standard ontologies).
*   `SCHEMA`: Refers to Schema.org, a widely used vocabulary for structured data on the internet. We'll try to map some of our extracted types to Schema.org terms for better interoperability.
*   `RDFS`: RDF Schema, provides basic vocabulary for describing RDF vocabularies (e.g., `rdfs:label`, `rdfs:Class`).
*   `RDF`: The core RDF vocabulary (e.g., `rdf:type`).
*   `XSD`: XML Schema Datatypes, used for specifying literal data types (e.g., `xsd:string`, `xsd:date`).
*   `SKOS`: Simple Knowledge Organization System, useful for thesauri, taxonomies, and controlled vocabularies (e.g., `skos:altLabel` for alternative names).
"""
logger.info("#### Define RDF Namespaces")

EX = Namespace("http://example.org/kg/")
SCHEMA = Namespace("http://schema.org/")

logger.debug(f"Custom namespace EX defined as: {EX}")
logger.debug(f"Schema.org namespace SCHEMA defined as: {SCHEMA}")

"""
**Output Explanation:**
This confirms the definition of our primary custom namespace (`EX`) and the `SCHEMA` namespace from Schema.org.

## Phase 1: Data Acquisition and Preparation
**(Ref: Ch. 1 – Big Data Ecosystem; Ch. 3 – Value Chain of Big Data Processing)**

**Theory (Phase Overview):**
This initial phase is critical in any data-driven project. It corresponds to the early stages of the Big Data value chain: "Data Acquisition" and parts of "Data Preparation/Preprocessing". The goal is to obtain the raw data and transform it into a state suitable for further processing and information extraction. Poor quality input data (the "Garbage In, Garbage Out" principle) will inevitably lead to a poor quality Knowledge Graph.

### Step 1.1: Data Acquisition
**Task:** Gather a collection of news articles.

**Book Concept:** (Ch. 1, Figures 1 & 2; Ch. 3 - Data Acquisition stage)
This step represents the "Data Sources" and "Ingestion" components of a Big Data ecosystem. We're tapping into an existing dataset (CNN/DailyMail via Hugging Face `datasets`) rather than scraping live news, but the principle is the same: bringing external data into our processing pipeline.

**Methodology:**
We will define a function `acquire_articles` to load the CNN/DailyMail dataset. To manage processing time and costs for this demonstration, and to focus on potentially relevant articles, this function will:
1.  Load a specified split (e.g., 'train') of the dataset.
2.  Optionally filter articles based on a list of keywords. For our goal of technology company acquisitions, keywords like "acquire", "merger", "technology", "startup" would be relevant. This is a simple heuristic; more advanced topic modeling or classification could be used for better filtering on larger datasets.
3.  Take a small sample of the (filtered) articles.

**Output:** A list of raw article data structures (typically dictionaries containing 'id', 'article' text, etc.).
"""
logger.info("## Phase 1: Data Acquisition and Preparation")

def acquire_articles(dataset_name="cnn_dailymail", version="3.0.0", split='train', sample_size=1000, keyword_filter=None):
    """Loads articles from the specified Hugging Face dataset, optionally filters them, and takes a sample."""
    logger.debug(f"Attempting to load dataset: {dataset_name} (version: {version}, split: '{split}')...")
    try:
        full_dataset = load_dataset(dataset_name, version, split=split, streaming=False) # Use streaming=False for easier slicing on smaller datasets
        logger.debug(f"Successfully loaded dataset. Total records in split: {len(full_dataset)}")
    except Exception as e:
        logger.debug(f"Error loading dataset {dataset_name}: {e}")
        logger.debug("Please ensure the dataset is available or you have internet connectivity.")
        return [] # Return empty list on failure

    raw_articles_list = []
    if keyword_filter:
        logger.debug(f"Filtering articles containing any of keywords: {keyword_filter}...")
        count = 0
        iteration_limit = min(len(full_dataset), sample_size * 20) # Look through at most 20x sample_size articles
        for i in tqdm(range(iteration_limit), desc="Filtering articles"):
            record = full_dataset[i]
            if any(keyword.lower() in record['article'].lower() for keyword in keyword_filter):
                raw_articles_list.append(record)
                count += 1
            if count >= sample_size:
                logger.debug(f"Found {sample_size} articles matching filter criteria within {i+1} records checked.")
                break
        if not raw_articles_list:
            logger.debug(f"Warning: No articles found with keywords {keyword_filter} within the first {iteration_limit} records. Returning an empty list.")
            return []
        raw_articles_list = raw_articles_list[:sample_size]
    else:
        logger.debug(f"Taking the first {sample_size} articles without keyword filtering.")
        actual_sample_size = min(sample_size, len(full_dataset))
        raw_articles_list = list(full_dataset.select(range(actual_sample_size)))

    logger.debug(f"Acquired {len(raw_articles_list)} articles.")
    return raw_articles_list

logger.debug("Function 'acquire_articles' defined.")

"""
**Output Explanation:**
This cell defines the `acquire_articles` function. It will print a confirmation once the function is defined in the Python interpreter's memory.

#### Execute Data Acquisition

**Theory:**
Now we call the `acquire_articles` function. We define keywords relevant to our goal (technology company acquisitions) to guide the filtering process. A `SAMPLE_SIZE` is set to keep the amount of data manageable for this demonstration. Smaller samples allow for faster iteration, especially when using LLMs which can have associated costs and latency.
"""
logger.info("#### Execute Data Acquisition")

ACQUISITION_KEYWORDS = ["acquire", "acquisition", "merger", "buyout", "purchased by", "acquired by", "takeover"]
TECH_KEYWORDS = ["technology", "software", "startup", "app", "platform", "digital", "AI", "cloud"]

FILTER_KEYWORDS = ACQUISITION_KEYWORDS

SAMPLE_SIZE = 10 # Keep very small for quick LLM processing in this demo notebook

raw_data_sample = []
raw_data_sample = acquire_articles(sample_size=SAMPLE_SIZE, keyword_filter=FILTER_KEYWORDS)

if raw_data_sample: # Check if the list is not empty
    logger.debug(f"\nExample of a raw acquired article (ID: {raw_data_sample[0]['id']}):")
    logger.debug(raw_data_sample[0]['article'][:500] + "...")
    logger.debug(f"\nNumber of fields in a record: {len(raw_data_sample[0].keys())}")
    logger.debug(f"Fields: {list(raw_data_sample[0].keys())}")
else:
    logger.debug("No articles were acquired. Subsequent steps involving article processing might be skipped or produce no output.")

"""
**Output Explanation:**
This block executes the data acquisition. It will print:
*   Messages about the data loading and filtering process.
*   The number of articles acquired.
*   A snippet of the first acquired article and its available fields, to verify the process and understand the data structure.

### Step 1.2: Data Cleaning & Preprocessing
**Task:** Perform basic text normalization.

**Book Concept:** (Ch. 3 - Variety challenge of Big Data)
Raw text data from sources like news articles is often messy. It can contain HTML tags, boilerplate content (like bylines, copyright notices), special characters, and inconsistent formatting. This step parallels addressing the "Variety" (and to some extent, "Veracity") challenge of Big Data. Clean, normalized input is crucial for effective downstream NLP tasks, as noise can significantly degrade the performance of entity recognizers and relation extractors.

**Methodology:**
We'll define a function `clean_article_text` that uses regular expressions (`re` module) to:
*   Remove common news boilerplate (e.g., "(CNN) --", specific byline patterns).
*   Remove HTML tags and URLs.
*   Normalize whitespace (e.g., replace multiple spaces/newlines with a single space).
*   Optionally, handle quotes or other special characters that might interfere with LLM processing or JSON formatting if not handled carefully.

**Output:** A list of dictionaries, where each dictionary contains the article ID and its cleaned text.
"""
logger.info("### Step 1.2: Data Cleaning & Preprocessing")

def clean_article_text(raw_text):
    """Cleans the raw text of a news article using regular expressions."""
    text = raw_text

    text = re.sub(r'^\(CNN\)\s*(--)?\s*', '', text)
    text = re.sub(r'By .*? for Dailymail\.com.*?Published:.*?Updated:.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'PUBLISHED:.*?BST,.*?UPDATED:.*?BST,.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'Last updated at.*on.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

logger.debug("Function 'clean_article_text' defined.")

"""
**Output Explanation:**
Confirms that the `clean_article_text` function, which will be used to preprocess article content, has been defined.

#### Execute Data Cleaning

**Theory:**
This block iterates through the `raw_data_sample` (acquired in the previous step). For each article, it calls the `clean_article_text` function. The cleaned text, along with the original article ID and potentially other useful fields like 'summary' (if available from the dataset as 'highlights'), is stored in a new list called `cleaned_articles`. This new list will be the primary input for the subsequent Information Extraction phase.
"""
logger.info("#### Execute Data Cleaning")

cleaned_articles = [] # Initialize as an empty list

if raw_data_sample: # Proceed only if raw_data_sample is not empty
    logger.debug(f"Cleaning {len(raw_data_sample)} acquired articles...")
    for record in tqdm(raw_data_sample, desc="Cleaning articles"):
        cleaned_text_content = clean_article_text(record['article'])
        cleaned_articles.append({
            "id": record['id'],
            "original_text": record['article'], # Keep original for reference
            "cleaned_text": cleaned_text_content,
            "summary": record.get('highlights', '') # CNN/DM has 'highlights' which are summaries
        })
    logger.debug(f"Finished cleaning. Total cleaned articles: {len(cleaned_articles)}.")
    if cleaned_articles: # Check if list is not empty after processing
        logger.debug(f"\nExample of a cleaned article (ID: {cleaned_articles[0]['id']}):")
        logger.debug(cleaned_articles[0]['cleaned_text'][:500] + "...")
else:
    logger.debug("No raw articles were acquired in the previous step, so skipping cleaning.")

if 'cleaned_articles' not in globals():
    cleaned_articles = []
    logger.debug("Initialized 'cleaned_articles' as an empty list because it was not created prior.")

"""
**Output Explanation:**
This block will:
*   Indicate the start and end of the cleaning process.
*   Show the number of articles cleaned.
*   Display a snippet of the first cleaned article's text, allowing for a visual check of the cleaning effectiveness.

## Phase 2: Information Extraction
**(Ref: Ch. 2 – Basics of Knowledge Graphs; Ch. 4 – KG Creation from Structured Data)**

**Theory (Phase Overview):**
Information Extraction (IE) is the process of automatically extracting structured information from unstructured or semi-structured sources (like our news articles). In the context of Knowledge Graph creation, IE is paramount as it identifies the fundamental building blocks: entities (nodes) and the relationships (edges) that connect them. This phase directly addresses how to transform raw text into a more structured format, a key step before KG materialization (Ch. 4). It involves tasks like Named Entity Recognition (NER) and Relationship Extraction (RE).

### Step 2.1: Entity Extraction (Named Entity Recognition - NER)
**Task:** Identify named entities like organizations, people, products, monetary figures, and dates.

**Book Concept:** (Ch. 2 - Entities as nodes)
Named Entities are real-world objects, such as persons, locations, organizations, products, etc., that can be denoted with a proper name. In a KG, these entities become the *nodes*. Accurate NER is foundational for building a meaningful graph.

**Methodology:**
We'll employ a two-pronged approach:
1.  **Exploratory NER with spaCy:** Use spaCy's pre-trained model to get a quick overview of common entity types present in our cleaned articles. This helps in understanding the general landscape of entities.
2.  **LLM-driven Entity Type Selection:** Based on spaCy's output and our specific goal (technology acquisitions), we'll prompt an LLM to suggest a focused set of entity types that are most relevant.
3.  **Targeted NER with LLM:** Use the LLM with the refined list of entity types to perform NER on the articles, aiming for higher accuracy and relevance for our specific domain. LLMs can be powerful here due to their contextual understanding, especially when guided by well-crafted prompts.

#### 2.1.1: Entity Exploration with spaCy - Function Definition

**Theory:**
This function, `get_spacy_entity_counts`, takes a list of articles, processes a sample of their text using spaCy's NER capabilities, and returns a counter object tallying the frequencies of different entity labels (e.g., `PERSON`, `ORG`, `GPE`). This gives us an empirical basis for understanding what kinds of entities are prevalent in our dataset before we engage the more resource-intensive LLM.
"""
logger.info("## Phase 2: Information Extraction")

def get_spacy_entity_counts(articles_data, text_field='cleaned_text', sample_size_spacy=50):
    """Processes a sample of articles with spaCy and counts entity labels."""
    if not nlp_spacy:
        logger.debug("spaCy model not loaded. Skipping spaCy entity counting.")
        return Counter()
    if not articles_data:
        logger.debug("No articles data provided to spaCy for entity counting. Skipping.")
        return Counter()

    label_counter = Counter()
    sample_to_process = articles_data[:min(len(articles_data), sample_size_spacy)]

    logger.debug(f"Processing {len(sample_to_process)} articles with spaCy for entity counts...")
    for article in tqdm(sample_to_process, desc="spaCy NER for counts"):
        doc = nlp_spacy(article[text_field])
        for ent in doc.ents:
            label_counter[ent.label_] += 1
    return label_counter

logger.debug("Function 'get_spacy_entity_counts' defined.")

"""
**Output Explanation:**
Confirms the definition of the `get_spacy_entity_counts` function.

#### 2.1.1: Entity Exploration with spaCy - Plotting Function Definition

**Theory:**
The `plot_entity_distribution` function takes the entity counts (from `get_spacy_entity_counts`) and uses `matplotlib` to generate a bar chart. Visualizing this distribution helps in quickly identifying the most frequent entity types, which can inform subsequent decisions about which types to prioritize for the KG.
"""
logger.info("#### 2.1.1: Entity Exploration with spaCy - Plotting Function Definition")

def plot_entity_distribution(label_counter_to_plot):
    """Plots the distribution of entity labels from a Counter object."""
    if not label_counter_to_plot:
        logger.debug("No entity counts to plot.")
        return

    top_items = label_counter_to_plot.most_common(min(15, len(label_counter_to_plot)))
    if not top_items: # Handle case where counter is not empty but most_common(0) or similar edge case
        logger.debug("No items to plot from entity counts.")
        return

    labels, counts = zip(*top_items)

    plt.figure(figsize=(12, 7))
    plt.bar(labels, counts, color='skyblue')
    plt.title("Top Entity Type Distribution (via spaCy)")
    plt.ylabel("Frequency")
    plt.xlabel("Entity Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout() # Adjust layout to make sure everything fits
    plt.show()

logger.debug("Function 'plot_entity_distribution' defined.")

"""
**Output Explanation:**
Confirms the definition of the `plot_entity_distribution` function.

#### 2.1.1: Entity Exploration with spaCy - Execution

**Theory:**
This block executes the spaCy-based entity exploration. It calls `get_spacy_entity_counts` on the `cleaned_articles`. The resulting counts are then printed and passed to `plot_entity_distribution` to visualize the findings. This step is skipped if no cleaned articles are available or if the spaCy model failed to load.
"""
logger.info("#### 2.1.1: Entity Exploration with spaCy - Execution")

spacy_entity_counts = Counter() # Initialize an empty counter

if cleaned_articles and nlp_spacy:
    spacy_analysis_sample_size = min(len(cleaned_articles), 20)
    logger.debug(f"Running spaCy NER on a sample of {spacy_analysis_sample_size} cleaned articles...")
    spacy_entity_counts = get_spacy_entity_counts(cleaned_articles, sample_size_spacy=spacy_analysis_sample_size)

    if spacy_entity_counts:
        logger.debug("\nspaCy Entity Counts (from sample):")
        for label, count in spacy_entity_counts.most_common():
            logger.debug(f"  {label}: {count}")
        plot_entity_distribution(spacy_entity_counts)
    else:
        logger.debug("spaCy NER did not return any entity counts from the sample.")
else:
    logger.debug("Skipping spaCy entity analysis: No cleaned articles available or spaCy model not loaded.")

"""
**Output Explanation:**
This block will print:
*   The frequency of different entity types found by spaCy in the sample.
*   A bar chart visualizing this distribution.
If prerequisites are not met, it will print a message indicating why this step was skipped.

#### Generic LLM Call Function Definition

**Theory:**
To interact with the LLM for various tasks (entity type selection, NER, relation extraction), we define a reusable helper function `call_llm_for_response`. This function encapsulates the logic for:
*   Taking a system prompt (instructions for the LLM) and a user prompt (the specific input/query).
*   Making the API call to the configured LLM endpoint.
*   Extracting the textual content from the LLM's response.
*   Basic error handling if the LLM client is not initialized or if the API call fails.
Using a helper function promotes code reusability and makes the main logic cleaner.
"""
logger.info("#### Generic LLM Call Function Definition")

def call_llm_for_response(system_prompt, user_prompt, model_to_use=TEXT_GEN_MODEL_NAME, temperature=0.2):
    """Generic function to call the LLM and get a response, with basic error handling."""
    if not client:
        logger.debug("LLM client not initialized. Skipping LLM call.")
        return "LLM_CLIENT_NOT_INITIALIZED"
    try:
        logger.debug(f"\nCalling LLM (model: {model_to_use}, temperature: {temperature})...")

        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature # Lower temperature for more focused/deterministic output
        )
        content = response.choices[0].message.content.strip()
        logger.debug("LLM response received.")
        return content
    except Exception as e:
        logger.debug(f"Error calling LLM: {e}")
        return f"LLM_ERROR: {str(e)}"

logger.debug("Function 'call_llm_for_response' defined.")

"""
**Output Explanation:**
Confirms the definition of the `call_llm_for_response` helper function.

#### 2.1.2: Entity Type Selection using LLM - Execution

**Theory:**
While spaCy provides a general set of entity types, not all may be relevant for our specific goal of building a KG about technology company acquisitions. For instance, `WORK_OF_ART` might be less important than `ORG` (organization) or `MONEY`. 
In this step, we leverage the LLM's understanding to refine this list. 
1.  **System Prompt:** We craft a detailed system prompt instructing the LLM to act as an expert in KG construction for technology news. It's asked to select the *most relevant* entity labels from the spaCy-derived list, focusing on our domain, and to provide an explanation for each chosen type.
2.  **User Prompt:** The user prompt contains the actual list of entity labels and their frequencies obtained from spaCy.
3.  **LLM Call:** We use our `call_llm_for_response` function.
The LLM's output should be a comma-separated string of chosen entity types with their descriptions (e.g., `ORG (Organizations involved in acquisitions, e.g., Google, Microsoft)`). This curated list forms a more targeted schema for our subsequent LLM-based NER.
"""
logger.info("#### 2.1.2: Entity Type Selection using LLM - Execution")

ENTITY_TYPE_SELECTION_SYSTEM_PROMPT = (
    "You are an expert assistant specializing in Knowledge Graph construction for technology news analysis. "
    "You will be provided with a list of named entity labels and their frequencies, derived from news articles. "
    "Your task is to select and return a comma-separated list of the MOST RELEVANT entity labels for building a Knowledge Graph focused on **technology company acquisitions**. "
    "Prioritize labels like organizations (acquirer, acquired), financial amounts (deal value), dates (announcement/closing), key persons (CEOs, founders), and relevant technology products/services or sectors. "
    "For EACH entity label you include in your output list, provide a concise parenthetical explanation or a clear, illustrative example. "
    "Example: ORG (Company involved in acquisition, e.g., Google, Microsoft), MONEY (Transaction value or investment, e.g., $1 billion), DATE (Date of acquisition announcement or closing, e.g., July 26, 2023). "
    "The output MUST be ONLY the comma-separated list of labels and their parenthetical explanations. "
    "Do not include any introductory phrases, greetings, summaries, or any other text whatsoever outside of this formatted list."
)

llm_selected_entity_types_str = "" # Initialize
DEFAULT_ENTITY_TYPES_STR = "ORG (Acquiring or acquired company, e.g., TechCorp), PERSON (Key executives, e.g., CEO), MONEY (Acquisition price, e.g., $500 million), DATE (Date of acquisition announcement), PRODUCT (Key product/service involved), GPE (Location of companies, e.g., Silicon Valley)"

if spacy_entity_counts and client: # Proceed if we have spaCy counts and LLM client is available
    spacy_labels_for_prompt = ", ".join([f"{label} (frequency: {count})" for label, count in spacy_entity_counts.most_common()])
    user_prompt_for_types = f"From the following entity labels and their frequencies found in news articles: [{spacy_labels_for_prompt}]. Please select and format the most relevant entity types for a knowledge graph about technology company acquisitions, as per the instructions."

    llm_selected_entity_types_str = call_llm_for_response(ENTITY_TYPE_SELECTION_SYSTEM_PROMPT, user_prompt_for_types)

    if "LLM_CLIENT_NOT_INITIALIZED" in llm_selected_entity_types_str or "LLM_ERROR" in llm_selected_entity_types_str or not llm_selected_entity_types_str.strip():
        logger.debug("\nLLM entity type selection failed or returned empty. Using default entity types.")
        llm_selected_entity_types_str = DEFAULT_ENTITY_TYPES_STR
    else:
        logger.debug("\nLLM Suggested Entity Types for Tech Acquisitions KG:")
        if not re.match(r"^([A-Z_]+ \(.*?\))(, [A-Z_]+ \(.*?\))*$", llm_selected_entity_types_str.strip()):
             logger.debug(f"Warning: LLM output for entity types might not be in the expected strict format. Raw: '{llm_selected_entity_types_str}'")
             lines = llm_selected_entity_types_str.strip().split('\n')
             best_line = ""
             for line in lines:
                 if '(' in line and ')' in line and len(line) > len(best_line):
                     best_line = line
             if best_line:
                 llm_selected_entity_types_str = best_line
                 logger.debug(f"Attempted cleanup: '{llm_selected_entity_types_str}'")
             else:
                 logger.debug("Cleanup failed, falling back to default entity types.")
                 llm_selected_entity_types_str = DEFAULT_ENTITY_TYPES_STR
else:
    logger.debug("\nSkipping LLM entity type selection (spaCy counts unavailable or LLM client not initialized). Using default entity types.")
    llm_selected_entity_types_str = DEFAULT_ENTITY_TYPES_STR

logger.debug(f"\nFinal list of Entity Types to be used for NER: {llm_selected_entity_types_str}")

"""
**Output Explanation:**
This block will print:
*   The comma-separated list of entity types and their descriptions as suggested by the LLM (or the default list if the LLM call fails/is skipped).
*   This list will guide the next step: targeted Named Entity Recognition.

#### LLM JSON Output Parsing Function Definition

**Theory:**
LLMs, even when prompted for specific formats like JSON, can sometimes produce output that includes extra text, markdown formatting (like ` ```json ... ``` `), or slight deviations from perfect JSON. The `parse_llm_json_output` function is a utility to robustly parse the LLM's string output into a Python list of dictionaries (representing entities or relations).
It attempts to:
1.  Handle common markdown code block syntax.
2.  Use `json.loads()` for parsing.
3.  Include error handling for `JSONDecodeError` and provide fallback mechanisms like regex-based extraction if simple parsing fails.
This function is crucial for reliably converting LLM responses into usable structured data.
"""
logger.info("#### LLM JSON Output Parsing Function Definition")

def parse_llm_json_output(llm_output_str):
    """Parses JSON output from LLM, handling potential markdown code blocks and common issues."""
    if not llm_output_str or "LLM_CLIENT_NOT_INITIALIZED" in llm_output_str or "LLM_ERROR" in llm_output_str:
        logger.debug("Cannot parse LLM output: LLM did not run, errored, or output was empty.")
        return [] # Return empty list

    match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_output_str, re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        list_start_index = llm_output_str.find('[')
        list_end_index = llm_output_str.rfind(']')
        if list_start_index != -1 and list_end_index != -1 and list_start_index < list_end_index:
            json_str = llm_output_str[list_start_index : list_end_index+1].strip()
        else:
            json_str = llm_output_str.strip() # Fallback to the whole string

    try:
        parsed_data = json.loads(json_str)
        if isinstance(parsed_data, list):
            return parsed_data
        else:
            logger.debug(f"Warning: LLM output was valid JSON but not a list (type: {type(parsed_data)}). Returning empty list.")
            logger.debug(f"Problematic JSON string (or part of it): {json_str[:200]}...")
            return []
    except json.JSONDecodeError as e:
        logger.debug(f"Error decoding JSON from LLM output: {e}")
        logger.debug(f"Problematic JSON string (or part of it): {json_str[:500]}...")
        return []
    except Exception as e:
        logger.debug(f"An unexpected error occurred during LLM JSON output parsing: {e}")
        return []

logger.debug("Function 'parse_llm_json_output' defined.")

"""
**Output Explanation:**
Confirms the definition of the `parse_llm_json_output` utility function.

#### 2.1.3: Targeted Entity Extraction using LLM - Execution

**Theory:**
Now, equipped with our curated list of entity types (`llm_selected_entity_types_str`), we instruct the LLM to perform NER on each (cleaned) article. 
1.  **System Prompt:** The system prompt for NER is carefully constructed. It tells the LLM:
    *   Its role (expert NER system for tech acquisitions).
    *   The specific entity types to focus on (from `llm_selected_entity_types_str`).
    *   The required output format: a JSON list of objects, where each object has `"text"` (the exact extracted entity span) and `"type"` (one of the specified entity types).
    *   An example of the desired JSON output.
    *   To output an empty JSON list `[]` if no relevant entities are found.
2.  **User Prompt:** For each article, the user prompt is simply its `cleaned_text`.
3.  **Processing Loop:** We iterate through a small sample of `cleaned_articles` (defined by `MAX_ARTICLES_FOR_LLM_NER` to manage time/cost). For each:
    *   The article text is (optionally truncated if too long for LLM context window).
    *   `call_llm_for_response` is invoked.
    *   `parse_llm_json_output` processes the LLM's response.
    *   The extracted entities are stored alongside the article data in a new list, `articles_with_entities`.
A small delay (`time.sleep`) is added between API calls to be polite to the API endpoint and avoid potential rate limiting.
"""
logger.info("#### 2.1.3: Targeted Entity Extraction using LLM - Execution")

LLM_NER_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert Named Entity Recognition system specialized in identifying information about **technology company acquisitions**. "
    "From the provided news article text, identify and extract entities. "
    "The entity types to focus on are: {entity_types_list_str}. "
    "Ensure that the extracted 'text' for each entity is an EXACT span from the article. "
    "Output ONLY a valid JSON list of objects, where each object has 'text' (the exact extracted entity string) and 'type' (one of the entity type main labels, e.g., ORG, PERSON, MONEY from your list) keys. "
    "Example: [{{'text': 'United Nations', 'type': 'ORG'}}, {{'text': 'Barack Obama', 'type': 'PERSON'}}, {{'text': 'iPhone 15', 'type': 'PRODUCT'}}]"
)

articles_with_entities = [] # Initialize
MAX_ARTICLES_FOR_LLM_NER = 3 # Process a very small number for this demo

if cleaned_articles and client and llm_selected_entity_types_str and "LLM_" not in llm_selected_entity_types_str:
    ner_system_prompt = LLM_NER_SYSTEM_PROMPT_TEMPLATE.format(entity_types_list_str=llm_selected_entity_types_str)

    num_articles_to_process_ner = min(len(cleaned_articles), MAX_ARTICLES_FOR_LLM_NER)
    logger.debug(f"Starting LLM NER for {num_articles_to_process_ner} articles...")

    for i, article_dict in enumerate(tqdm(cleaned_articles[:num_articles_to_process_ner], desc="LLM NER Processing")):
        logger.debug(f"\nProcessing article ID: {article_dict['id']} for NER with LLM ({i+1}/{num_articles_to_process_ner})...")

        max_text_chars = 12000 # Approx 3000 words. Should be safe for many models.
        article_text_for_llm = article_dict['cleaned_text'][:max_text_chars]
        if len(article_dict['cleaned_text']) > max_text_chars:
            logger.debug(f"  Warning: Article text truncated from {len(article_dict['cleaned_text'])} to {max_text_chars} characters for LLM NER.")

        llm_ner_raw_output = call_llm_for_response(ner_system_prompt, article_text_for_llm)
        extracted_entities_list = parse_llm_json_output(llm_ner_raw_output)

        current_article_data = article_dict.copy() # Make a copy to avoid modifying original list items directly
        current_article_data['llm_entities'] = extracted_entities_list
        articles_with_entities.append(current_article_data)

        logger.debug(f"  Extracted {len(extracted_entities_list)} entities for article ID {article_dict['id']}.")
        if extracted_entities_list:
            logger.debug(f"  Sample entities: {json.dumps(extracted_entities_list[:min(3, len(extracted_entities_list))], indent=2)}")

        if i < num_articles_to_process_ner - 1: # Avoid sleeping after the last article
            time.sleep(1) # Small delay to be polite to API

    if articles_with_entities:
        logger.debug(f"\nFinished LLM NER. Processed {len(articles_with_entities)} articles and stored entities.")
else:
    logger.debug("Skipping LLM NER: Prerequisites (cleaned articles, LLM client, or valid entity types string) not met.")
    if cleaned_articles: # only if we have cleaned articles to begin with
        num_articles_to_fallback = min(len(cleaned_articles), MAX_ARTICLES_FOR_LLM_NER)
        for article_dict_fallback in cleaned_articles[:num_articles_to_fallback]:
            fallback_data = article_dict_fallback.copy()
            fallback_data['llm_entities'] = []
            articles_with_entities.append(fallback_data)
        logger.debug(f"Populated 'articles_with_entities' with {len(articles_with_entities)} entries having empty 'llm_entities' lists.")

if 'articles_with_entities' not in globals():
    articles_with_entities = []
    logger.debug("Initialized 'articles_with_entities' as an empty list.")

"""
**Output Explanation:**
This block will show the progress of LLM-based NER:
*   For each processed article: its ID, a message about truncation (if any), the number of entities extracted, and a sample of the extracted entities in JSON format.
*   A final message indicating completion or why the step was skipped.
The `articles_with_entities` list now contains the original article data plus a new key `llm_entities` holding the list of entities extracted by the LLM for that article.

### Step 2.2: Relationship Extraction
**Task:** Identify semantic relationships between extracted entities, such as acquisition events or affiliations.

**Book Concept:** (Ch. 2 - Relationships as edges)
Relationships define how entities are connected, forming the *edges* in our Knowledge Graph. Extracting these relationships is crucial for capturing the actual knowledge (e.g., "Company A *acquired* Company B", "Acquisition *has_price* $X Million").

**Methodology:**
Similar to NER, we'll use the LLM for Relationship Extraction (RE). For each article:
1.  **System Prompt:** We design a system prompt that instructs the LLM to act as a relationship extraction expert for technology acquisitions. It specifies:
    *   The desired relationship types (predicates) like `ACQUIRED`, `HAS_PRICE`, `ANNOUNCED_ON`, `ACQUIRER_IS`, `ACQUIRED_COMPANY_IS`, `INVOLVES_PRODUCT`. These are chosen to capture key aspects of an acquisition event.
    *   The requirement that the subject and object of a relationship must be exact text spans from the list of entities provided for that article.
    *   The desired output format: a JSON list of objects, each with `subject_text`, `subject_type`, `predicate`, `object_text`, and `object_type`.
    *   An example of the output format.
2.  **User Prompt:** The user prompt for each article will contain:
    *   The article's `cleaned_text`.
    *   The list of `llm_entities` extracted in the previous step for that specific article (serialized as a JSON string within the prompt).
3.  **Processing Loop:** Iterate through `articles_with_entities`. If an article has entities:
    *   Construct the user prompt.
    *   Call the LLM.
    *   Parse the JSON output.
    *   Optionally, validate that the subject/object texts in the extracted relations indeed come from the provided entity list to maintain consistency.
    *   Store the extracted relations in a new list, `articles_with_relations` (each item will be the article data augmented with `llm_relations`).
"""
logger.info("### Step 2.2: Relationship Extraction")

RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert system for extracting relationships between entities from text, specifically focusing on **technology company acquisitions**. "
    "Example: [{{'subject_text': 'Innovatech Ltd.', 'subject_type': 'ORG', 'predicate': 'ACQUIRED', 'object_text': 'Global Solutions Inc.', 'object_type': 'ORG'}}, {{'subject_text': 'Global Solutions Inc.', 'subject_type': 'ORG', 'predicate': 'HAS_PRICE', 'object_text': '$250M', 'object_type': 'MONEY'}}] "
    "If no relevant relationships of the specified types are found between the provided entities, output an empty JSON list []. Do not output any other text or explanation."
)

articles_with_relations = [] # Initialize

if articles_with_entities and client: # Proceed if we have entities and LLM client
    relation_system_prompt = RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT_TEMPLATE # System prompt is constant for this task

    logger.debug(f"Starting LLM Relationship Extraction for {len(articles_with_entities)} articles (that have entities)...")
    for i, article_data_with_ents in enumerate(tqdm(articles_with_entities, desc="LLM Relationship Extraction")):
        logger.debug(f"\nProcessing article ID: {article_data_with_ents['id']} for relationships ({i+1}/{len(articles_with_entities)})...")
        article_text_content = article_data_with_ents['cleaned_text']
        entities_for_article = article_data_with_ents['llm_entities']

        if not entities_for_article: # If an article had no entities extracted
            logger.debug(f"  No entities found for article ID {article_data_with_ents['id']}, skipping relationship extraction for this article.")
            current_article_output = article_data_with_ents.copy()
            current_article_output['llm_relations'] = []
            articles_with_relations.append(current_article_output)
            continue

        max_text_chars_re = 10000 # Slightly less than NER to accommodate entity list in prompt
        article_text_for_llm_re = article_text_content[:max_text_chars_re]
        if len(article_text_content) > max_text_chars_re:
            logger.debug(f"  Warning: Article text truncated from {len(article_text_content)} to {max_text_chars_re} characters for LLM RE.")

        entities_for_prompt_str = json.dumps([{'text': e['text'], 'type': e['type']} for e in entities_for_article])

        user_prompt_for_relations = (
            f"Article Text:\n```\n{article_text_for_llm_re}\n```\n\n"
            f"Extracted Entities (use these exact texts for subjects/objects):\n```json\n{entities_for_prompt_str}\n```\n\n"
            f"Identify and extract relationships between these entities based on the system instructions."
        )

        llm_relations_raw_output = call_llm_for_response(relation_system_prompt, user_prompt_for_relations)
        extracted_relations_list = parse_llm_json_output(llm_relations_raw_output)

        valid_relations_list = []
        entity_texts_in_article = {e['text'] for e in entities_for_article}
        for rel in extracted_relations_list:
            if isinstance(rel, dict) and rel.get('subject_text') in entity_texts_in_article and rel.get('object_text') in entity_texts_in_article:
                valid_relations_list.append(rel)
            else:
                logger.debug(f"  Warning: Discarding relation due to missing fields or mismatched entity text: {str(rel)[:100]}...")

        current_article_output = article_data_with_ents.copy()
        current_article_output['llm_relations'] = valid_relations_list
        articles_with_relations.append(current_article_output)

        logger.debug(f"  Extracted {len(valid_relations_list)} valid relationships for article ID {article_data_with_ents['id']}.")
        if valid_relations_list:
            logger.debug(f"  Sample relations: {json.dumps(valid_relations_list[:min(2, len(valid_relations_list))], indent=2)}")

        if i < len(articles_with_entities) - 1:
             time.sleep(1) # Small delay

    if articles_with_relations:
        logger.debug(f"\nFinished LLM Relationship Extraction. Processed {len(articles_with_relations)} articles and stored relations.")
else:
    logger.debug("Skipping LLM Relationship Extraction: Prerequisites (articles with entities, LLM client) not met.")
    if articles_with_entities: # only if we have articles from NER step
        for article_data_fallback_re in articles_with_entities:
            fallback_data_re = article_data_fallback_re.copy()
            fallback_data_re['llm_relations'] = []
            articles_with_relations.append(fallback_data_re)
        logger.debug(f"Populated 'articles_with_relations' with {len(articles_with_relations)} entries having empty 'llm_relations' lists.")

if 'articles_with_relations' not in globals():
    articles_with_relations = []
    logger.debug("Initialized 'articles_with_relations' as an empty list.")

"""
**Output Explanation:**
This block will show the progress of LLM-based Relationship Extraction:
*   For each processed article: its ID, number of relationships extracted, and a sample of these relations in JSON format.
*   Warnings if any extracted relations are discarded due to validation failures.
*   A final message indicating completion or why the step was skipped.
The `articles_with_relations` list now contains items that have `llm_entities` and `llm_relations`.

## Phase 3: Knowledge Graph Construction
**(Ref: Ch. 2 – KG Layers; Ch. 4 – Mapping and Materialization)**

**Theory (Phase Overview):**
Having extracted entities and relationships, this phase focuses on formally constructing the Knowledge Graph. This involves several key sub-tasks:
*   **Entity Disambiguation & Linking:** Ensuring that different textual mentions of the same real-world entity are resolved to a single, canonical identifier (URI). This is crucial for graph coherence and data integration (Ch. 6, Ch. 8 concepts like COMET).
*   **Schema/Ontology Alignment:** Mapping the extracted entity types and relationship predicates to a predefined schema or ontology (Ch. 2 - Ontology Layer; Ch. 4 - R2RML-like mapping). This provides semantic structure and enables interoperability and reasoning.
*   **Triple Generation:** Converting the structured entity and relation data into Subject-Predicate-Object (S-P-O) triples, the fundamental data model of RDF-based KGs (Ch. 2, Ch. 4 - RML output).
The output of this phase is a populated `rdflib.Graph` object.

### Step 3.1: Entity Disambiguation & Linking (Simplified) - Normalization Function
**Task:** Resolve different mentions of the same real-world entity.

**Book Concept:** (Ch. 6 - Entity Resolution; Ch. 8 - Context-aware linking)
True entity disambiguation and linking (EDL) is a complex NLP task, often involving linking entities to large external KGs like Wikidata or DBpedia, or using sophisticated clustering and coreference resolution. 

**Methodology (Simplified):**
For this demonstration, we'll perform a simplified version: **text normalization**. The `normalize_entity_text` function will:
*   Trim whitespace.
*   For `ORG` entities, attempt to remove common corporate suffixes (e.g., "Inc.", "Ltd.", "Corp.") to group variations like "Example Corp" and "Example Corporation" under a common normalized form like "Example".
*   (Optionally, one might consider lowercasing, but this can sometimes lose important distinctions, e.g., between "IT" the pronoun and "IT" the sector).
This normalized text will then be used to create a unique URI for each distinct entity.
"""
logger.info("## Phase 3: Knowledge Graph Construction")

def normalize_entity_text(text_to_normalize, entity_type_str):
    """Normalizes entity text for better linking (simplified version)."""
    normalized_t = text_to_normalize.strip()

    if entity_type_str == 'ORG':
        suffixes = [
            'Inc.', 'Incorporated', 'Ltd.', 'Limited', 'LLC', 'L.L.C.',
            'Corp.', 'Corporation', 'PLC', 'Public Limited Company',
            'GmbH', 'AG', 'S.A.', 'S.A.S.', 'B.V.', 'Pty Ltd', 'Co.', 'Company',
            'Solutions', 'Technologies', 'Systems', 'Group', 'Holdings'
        ]
        suffixes.sort(key=len, reverse=True)
        for suffix in suffixes:
            if normalized_t.lower().endswith(suffix.lower()):
                suffix_start_index = normalized_t.lower().rfind(suffix.lower())
                normalized_t = normalized_t[:suffix_start_index].strip()
                break # Remove one suffix type, then re-evaluate if more complex logic is needed

        normalized_t = re.sub(r'[-,.]*$', '', normalized_t).strip()

    if normalized_t.endswith("'s") or normalized_t.endswith("s'"):
        normalized_t = normalized_t[:-2].strip()


    return normalized_t.strip() if normalized_t else text_to_normalize # Return original if normalization results in empty

logger.debug("Function 'normalize_entity_text' defined.")

"""
**Output Explanation:**
Confirms the definition of the `normalize_entity_text` function.

#### Execute Entity Normalization and URI Generation

**Theory:**
This block processes the `articles_with_relations` list. For each entity extracted by the LLM:
1.  Its text is normalized using `normalize_entity_text`.
2.  A unique URI (Uniform Resource Identifier) is generated for each distinct normalized entity. We use a simple scheme: `EX:<NormalizedText>_<EntityType>`. The `EX` is our custom namespace. This creates a canonical identifier for each unique real-world concept (as per our normalization). A `unique_entities_map` dictionary stores the mapping from `(normalized_text, entity_type)` to its URI, ensuring that the same normalized entity always gets the same URI across all articles.
3.  The normalized text and the URI are added to the entity's dictionary.
The results are stored in `articles_with_normalized_entities`.
"""
logger.info("#### Execute Entity Normalization and URI Generation")

articles_with_normalized_entities = [] # Initialize
unique_entities_map = {} # Maps (normalized_text, type) -> URI, to ensure URI consistency

if articles_with_relations: # Process only if we have articles with relations (which implies entities)
    logger.debug("Normalizing entities and preparing for triple generation...")
    for article_data_rel in tqdm(articles_with_relations, desc="Normalizing Entities & URI Gen"):
        current_article_normalized_ents = []
        if 'llm_entities' in article_data_rel and isinstance(article_data_rel['llm_entities'], list):
            for entity_dict in article_data_rel['llm_entities']:
                if not (isinstance(entity_dict, dict) and 'text' in entity_dict and 'type' in entity_dict):
                    logger.debug(f"  Skipping malformed entity object: {str(entity_dict)[:100]} in article {article_data_rel['id']}")
                    continue

                original_entity_text = entity_dict['text']
                entity_type_val = entity_dict['type']
                simple_entity_type = entity_type_val.split(' ')[0].upper()
                entity_dict['simple_type'] = simple_entity_type # Store the simplified type

                normalized_entity_text = normalize_entity_text(original_entity_text, simple_entity_type)
                if not normalized_entity_text: # if normalization resulted in empty string, use original
                    normalized_entity_text = original_entity_text

                entity_map_key = (normalized_entity_text, simple_entity_type)

                if entity_map_key not in unique_entities_map:
                    safe_uri_text_part = re.sub(r'[^a-zA-Z0-9_\-]', '_', normalized_entity_text.replace(' ', '_'))
                    safe_uri_text_part = safe_uri_text_part[:80]
                    if not safe_uri_text_part: # If sanitization results in empty string, use a hash or generic id
                        safe_uri_text_part = f"entity_{hashlib.md5(normalized_entity_text.encode()).hexdigest()[:8]}"
                    unique_entities_map[entity_map_key] = EX[f"{safe_uri_text_part}_{simple_entity_type}"]

                entity_dict_copy = entity_dict.copy()
                entity_dict_copy['normalized_text'] = normalized_entity_text
                entity_dict_copy['uri'] = unique_entities_map[entity_map_key]
                current_article_normalized_ents.append(entity_dict_copy)

        article_data_output_norm = article_data_rel.copy()
        article_data_output_norm['normalized_entities'] = current_article_normalized_ents
        articles_with_normalized_entities.append(article_data_output_norm)

    if articles_with_normalized_entities and articles_with_normalized_entities[0].get('normalized_entities'):
        logger.debug("\nExample of first article's normalized entities (first 3):")
        for ent_example in articles_with_normalized_entities[0]['normalized_entities'][:3]:
            logger.debug(f"  Original: '{ent_example['text']}', Type: {ent_example['type']} (Simple: {ent_example['simple_type']}), Normalized: '{ent_example['normalized_text']}', URI: <{ent_example['uri']}>")
    logger.debug(f"\nProcessed {len(articles_with_normalized_entities)} articles for entity normalization and URI generation.")
    logger.debug(f"Total unique canonical entity URIs created: {len(unique_entities_map)}")
else:
    logger.debug("Skipping entity normalization and URI generation: No articles with relations available.")

if 'articles_with_normalized_entities' not in globals():
    articles_with_normalized_entities = []
    logger.debug("Initialized 'articles_with_normalized_entities' as an empty list.")

"""
**Output Explanation:**
This block will show:
*   Progress of the normalization and URI generation process.
*   Examples of original entity text vs. their normalized versions and the generated URIs for the first few entities in the first processed article.
*   The total count of unique entity URIs created across all processed articles.

### Step 3.2: Schema/Ontology Alignment - RDF Type Mapping Function
**Task:** Map extracted entities and relationships to a consistent schema or ontology.

**Book Concept:** (Ch. 2 - Ontology Layer; Ch. 4 - Mapping)
Schema/Ontology alignment involves mapping our locally defined entity types (e.g., "ORG", "PERSON" from the LLM NER step) and relationship predicates to standard vocabularies (like Schema.org) or custom-defined RDF classes and properties. This adds semantic rigor and enables interoperability.

**Methodology:**
The `get_rdf_type_for_entity` function takes our simple entity type string (e.g., "ORG") and maps it to an RDF Class. 
*   It uses a predefined dictionary (`type_mapping`) to link common types to `SCHEMA` (Schema.org) classes (e.g., `ORG` -> `SCHEMA.Organization`).
*   If a type is not in the map, it defaults to creating a class within our custom `EX` namespace (e.g., `EX.CUSTOM_TYPE`).
This function ensures that each entity in our KG will be assigned a formal RDF type.
"""
logger.info("### Step 3.2: Schema/Ontology Alignment - RDF Type Mapping Function")

def get_rdf_type_for_entity(simple_entity_type_str):
    """Maps our simple entity type string (e.g., 'ORG') to an RDF Class."""
    type_mapping = {
        'ORG': SCHEMA.Organization,
        'PERSON': SCHEMA.Person,
        'MONEY': SCHEMA.PriceSpecification, # Or a custom EX.MonetaryValue
        'DATE': SCHEMA.Date, # Note: schema.org/Date is a datatype, consider schema.org/Event for events with dates
        'PRODUCT': SCHEMA.Product,
        'GPE': SCHEMA.Place,    # Geopolitical Entity
        'LOC': SCHEMA.Place,    # General Location
        'EVENT': SCHEMA.Event,
        'CARDINAL': RDF.Statement, # Or more specific if context known, often just a literal value
        'FAC': SCHEMA.Place # Facility
    }
    return type_mapping.get(simple_entity_type_str.upper(), EX[simple_entity_type_str.upper()]) # Fallback to custom type

logger.debug("Function 'get_rdf_type_for_entity' defined.")

"""
**Output Explanation:**
Confirms the definition of the `get_rdf_type_for_entity` mapping function.

#### Schema/Ontology Alignment - RDF Predicate Mapping Function

**Theory:**
The `get_rdf_predicate` function maps our string-based relationship predicates (e.g., "ACQUIRED", "HAS_PRICE" from the LLM RE step) to RDF Properties. For simplicity and custom control, these are typically mapped to properties within our `EX` namespace. The function ensures that predicate strings are converted into valid URI components (e.g., by replacing spaces with underscores).
"""
logger.info("#### Schema/Ontology Alignment - RDF Predicate Mapping Function")

def get_rdf_predicate(predicate_str_from_llm):
    """Maps our predicate string (from LLM relation extraction) to an RDF Property in our EX namespace."""
    sanitized_predicate = predicate_str_from_llm.strip().replace(" ", "_").upper()
    return EX[sanitized_predicate]

logger.debug("Function 'get_rdf_predicate' defined.")

"""
**Output Explanation:**
Confirms the definition of the `get_rdf_predicate` mapping function.

#### Schema/Ontology Alignment - Examples

**Theory:**
This block simply prints out a few examples of how our entity types and relationship predicates would be mapped to RDF terms using the functions defined above. This serves as a quick check and illustration of the mapping logic.
"""
logger.info("#### Schema/Ontology Alignment - Examples")

logger.debug("Schema alignment functions ready. Example mappings:")
example_entity_type = 'ORG'
example_predicate_str = 'ACQUIRED'
logger.debug(f"  Entity Type '{example_entity_type}' maps to RDF Class: <{get_rdf_type_for_entity(example_entity_type)}>")
logger.debug(f"  Predicate '{example_predicate_str}' maps to RDF Property: <{get_rdf_predicate(example_predicate_str)}>")

example_entity_type_2 = 'MONEY'
example_predicate_str_2 = 'HAS_PRICE'
logger.debug(f"  Entity Type '{example_entity_type_2}' maps to RDF Class: <{get_rdf_type_for_entity(example_entity_type_2)}>")
logger.debug(f"  Predicate '{example_predicate_str_2}' maps to RDF Property: <{get_rdf_predicate(example_predicate_str_2)}>")

"""
**Output Explanation:**
Shows example RDF URIs that would be generated for sample entity types and predicate strings, illustrating the mapping functions.

### Step 3.3: Triple Generation
**Task:** Convert the structured entity and relation data into subject–predicate–object triples.

**Book Concept:** (Ch. 2 - KG structure; Ch. 4 - RML output)
This is where the Knowledge Graph materializes. We iterate through our processed articles (`articles_with_normalized_entities`) and convert the extracted information into RDF triples using `rdflib`.

**Methodology:**
1.  **Initialize Graph:** An `rdflib.Graph` object (`kg`) is created.
2.  **Bind Namespaces:** Namespaces (EX, SCHEMA, RDFS, etc.) are bound to prefixes for cleaner serialization of the RDF (e.g., `ex:AcmeCorp` instead of the full URI).
3.  **Iterate Articles:** For each article:
    *   An RDF resource is created for the article itself (e.g., `ex:article_123`), typed as `schema:Article`.
    *   Its summary (or ID) can be added as a `schema:headline` or `rdfs:label`.
4.  **Iterate Entities:** For each `normalized_entity` within an article:
    *   The entity's URI (from `entity['uri']`) is used as the subject.
    *   A triple `(entity_uri, rdf:type, rdf_entity_type)` is added, where `rdf_entity_type` comes from `get_rdf_type_for_entity()`.
    *   A triple `(entity_uri, rdfs:label, Literal(normalized_text))` is added to provide a human-readable label.
    *   If the original text differs from the normalized text, `(entity_uri, skos:altLabel, Literal(original_text))` can be added for the original mention.
    *   A triple `(article_uri, schema:mentions, entity_uri)` links the article to the entities it mentions.
    *   A local map (`entity_uri_map_for_article`) is built for the current article, mapping original entity texts to their canonical URIs. This is crucial for resolving relations in the next step, as relations were extracted based on original text spans.
5.  **Iterate Relations:** For each `llm_relation` within an article:
    *   The URIs for the subject and object entities are looked up from `entity_uri_map_for_article` using their original text spans.
    *   The predicate string is converted to an RDF property using `get_rdf_predicate()`.
    *   If both subject and object URIs are found, the triple `(subject_uri, predicate_rdf, object_uri)` is added to the graph.

**Output:** A populated `rdflib.Graph` (`kg`) containing all the extracted knowledge as RDF triples.
"""
logger.info("### Step 3.3: Triple Generation")

kg = Graph() # Initialize an empty RDF graph
kg.bind("ex", EX)
kg.bind("schema", SCHEMA)
kg.bind("rdf", RDF)
kg.bind("rdfs", RDFS)
kg.bind("xsd", XSD)
kg.bind("skos", SKOS)

triples_generated_count = 0

if articles_with_normalized_entities:
    logger.debug(f"Generating RDF triples for {len(articles_with_normalized_entities)} processed articles...")
    for article_data_final in tqdm(articles_with_normalized_entities, desc="Generating Triples"):
        article_uri = EX[f"article_{article_data_final['id'].replace('-', '_')}"] # Sanitize ID for URI
        kg.add((article_uri, RDF.type, SCHEMA.Article))
        kg.add((article_uri, SCHEMA.headline, Literal(article_data_final.get('summary', article_data_final['id']))))
        triples_generated_count += 2

        entity_text_to_uri_map_current_article = {}

        for entity_obj in article_data_final.get('normalized_entities', []):
            entity_uri_val = entity_obj['uri'] # This is the canonical URI from unique_entities_map
            rdf_entity_type_val = get_rdf_type_for_entity(entity_obj['simple_type'])
            normalized_label = entity_obj['normalized_text']
            original_label = entity_obj['text']

            kg.add((entity_uri_val, RDF.type, rdf_entity_type_val))
            kg.add((entity_uri_val, RDFS.label, Literal(normalized_label, lang='en')))
            triples_generated_count += 2
            if normalized_label != original_label:
                 kg.add((entity_uri_val, SKOS.altLabel, Literal(original_label, lang='en')))
                 triples_generated_count += 1

            kg.add((article_uri, SCHEMA.mentions, entity_uri_val))
            triples_generated_count += 1

            entity_text_to_uri_map_current_article[original_label] = entity_uri_val

        for relation_obj in article_data_final.get('llm_relations', []):
            subject_orig_text = relation_obj.get('subject_text')
            object_orig_text = relation_obj.get('object_text')
            predicate_str = relation_obj.get('predicate')

            subject_resolved_uri = entity_text_to_uri_map_current_article.get(subject_orig_text)
            object_resolved_uri = entity_text_to_uri_map_current_article.get(object_orig_text)

            if subject_resolved_uri and object_resolved_uri and predicate_str:
                predicate_rdf_prop = get_rdf_predicate(predicate_str)
                kg.add((subject_resolved_uri, predicate_rdf_prop, object_resolved_uri))
                triples_generated_count += 1
            else:
                if not subject_resolved_uri:
                    logger.debug(f"  Warning: Could not find URI for subject '{subject_orig_text}' in article {article_data_final['id']}. Relation skipped: {relation_obj}")
                if not object_resolved_uri:
                    logger.debug(f"  Warning: Could not find URI for object '{object_orig_text}' in article {article_data_final['id']}. Relation skipped: {relation_obj}")

    logger.debug(f"\nFinished generating triples. Approximately {triples_generated_count} triples were candidates for addition.")
    logger.debug(f"Total actual triples in the graph: {len(kg)}")
    if len(kg) > 0:
        logger.debug("\nSample of first 5 triples in N3 format:")
        for i, (s, p, o) in enumerate(kg):
            logger.debug(f"  {s.n3(kg.namespace_manager)} {p.n3(kg.namespace_manager)} {o.n3(kg.namespace_manager)}.")
            if i >= 4: # Print first 5
                break
else:
    logger.debug("Skipping triple generation: No processed articles with normalized entities available.")

if 'kg' not in globals():
    kg = Graph()
    logger.debug("Initialized 'kg' as an empty rdflib.Graph object.")

"""
**Output Explanation:**
This block will show:
*   Progress of triple generation.
*   The approximate number of triples considered for addition and the final total number of triples in the `kg` graph.
*   Warnings if subject/object entities for a relation couldn't be resolved to URIs.
*   A sample of the first few triples in N3 (Notation3) format, which is a human-readable RDF serialization.

## Phase 4: Knowledge Graph Refinement Using Embeddings
**(Ref: Ch. 6 – Embedding-Based Reasoning; Ch. 7 – ML on KGs with SANSA)**

**Theory (Phase Overview):**
Knowledge Graph embeddings (KGEs) learn low-dimensional vector representations for entities and relations in a KG. These embeddings capture the semantic properties of KG components and their interactions. This phase explores using such embeddings for KG refinement, a concept aligned with embedding-based reasoning (Ch. 6).
Key tasks include:
*   **Generating Embeddings:** Creating vector representations for nodes (entities).
*   **Link Prediction (Knowledge Discovery):** Using embeddings to infer missing connections or predict new potential relationships (Ch. 6). This is a powerful way to discover implicit knowledge and enrich the KG.
While full KGE model training (like TransE, ComplEx, DistMult mentioned in Ch. 6) is beyond this demo's scope, we'll use pre-trained text embeddings for entity names as a proxy to demonstrate semantic similarity, a foundational concept for some link prediction approaches.

### Step 4.1: Generate KG Embeddings - Embedding Function Definition
**Task:** Create vector representations for nodes (entities) in the graph.

**Book Concept:** (Ch. 6 - Embeddings bridging symbolic & sub-symbolic)
Embeddings transform symbolic entities (represented by URIs and labels) into numerical vectors in a continuous vector space. This allows us to apply machine learning techniques and measure semantic similarity.

**Methodology:**
The `get_embeddings_for_texts` function will:
*   Take a list of unique entity texts (e.g., their normalized labels).
*   Use the configured MLX/Nebius embedding API (with `EMBEDDING_MODEL_NAME`) to fetch pre-trained embeddings for these texts.
*   Handle batching or individual requests as appropriate for the API.
*   Return a dictionary mapping each input text to its embedding vector.
These embeddings represent the semantic meaning of the entity names.
"""
logger.info("## Phase 4: Knowledge Graph Refinement Using Embeddings")

def get_embeddings_for_texts(texts_list, embedding_model_name=EMBEDDING_MODEL_NAME):
    """Gets embeddings for a list of texts using the specified model via the LLM client."""
    if not client:
        logger.debug("LLM client not initialized. Skipping embedding generation.")
        return {text: [] for text in texts_list} # Return dict with empty embeddings
    if not texts_list:
        logger.debug("No texts provided for embedding generation.")
        return {}

    embeddings_map_dict = {}
    logger.debug(f"Fetching embeddings for {len(texts_list)} unique texts using model '{embedding_model_name}'...")


    if not all(isinstance(text, str) for text in texts_list):
        logger.debug("Error: Input 'texts_list' must be a list of strings.")
        return {text: [] for text in texts_list if isinstance(text, str)} # Try to salvage what we can

    valid_texts_list = [text for text in texts_list if text.strip()]
    if not valid_texts_list:
        logger.debug("No valid (non-empty) texts to embed.")
        return {}

    try:
        response = client.embeddings.create(
            model=embedding_model_name,
            input=valid_texts_list # Pass the list of valid texts
        )
        for i, data_item in enumerate(response.data):
            embeddings_map_dict[valid_texts_list[i]] = data_item.embedding

        logger.debug(f"Embeddings received for {len(embeddings_map_dict)} texts.")
        for text in texts_list:
            if text not in embeddings_map_dict:
                embeddings_map_dict[text] = []
        return embeddings_map_dict

    except Exception as e:
        logger.debug(f"Error getting embeddings (batch attempt): {e}")
        logger.debug("Falling back to individual embedding requests if batch failed...")
        embeddings_map_dict_fallback = {}
        for text_input_item in tqdm(valid_texts_list, desc="Generating Embeddings (Fallback Mode)"):
            try:
                response_item = client.embeddings.create(
                    model=embedding_model_name,
                    input=text_input_item
                )
                embeddings_map_dict_fallback[text_input_item] = response_item.data[0].embedding
                if len(valid_texts_list) > 10: # Only sleep if processing many items
                    time.sleep(0.1) # Small delay per request in fallback
            except Exception as e_item:
                logger.debug(f"  Error getting embedding for text '{text_input_item[:50]}...': {e_item}")
                embeddings_map_dict_fallback[text_input_item] = [] # Store empty list on error for this item

        for text in texts_list:
            if text not in embeddings_map_dict_fallback:
                embeddings_map_dict_fallback[text] = []
        return embeddings_map_dict_fallback

logger.debug("Function 'get_embeddings_for_texts' defined.")

"""
**Output Explanation:**
Confirms the definition of the `get_embeddings_for_texts` function.

#### Generate KG Embeddings - Execution

**Theory:**
This block orchestrates the generation of embeddings for our KG entities:
1.  It extracts the set of unique, normalized entity texts from our `unique_entities_map` (which maps `(normalized_text, type)` to URIs). We are interested in embedding the textual representation of entities.
2.  It calls `get_embeddings_for_texts` with this list of unique texts.
3.  The returned embeddings (which are mapped to texts) are then re-mapped to our canonical entity URIs, creating the `entity_embeddings` dictionary: `{entity_uri: embedding_vector}`.
This dictionary will store the vector representation for each unique entity in our graph.
"""
logger.info("#### Generate KG Embeddings - Execution")

entity_embeddings = {} # Initialize: maps entity URI -> embedding vector

if unique_entities_map and client: # Proceed if we have unique entities and LLM client
    entity_normalized_texts_to_embed = list(set([key[0] for key in unique_entities_map.keys() if key[0].strip()]))

    if entity_normalized_texts_to_embed:
        logger.debug(f"Preparing to fetch embeddings for {len(entity_normalized_texts_to_embed)} unique normalized entity texts.")

        text_to_embedding_map = get_embeddings_for_texts(entity_normalized_texts_to_embed)

        for (normalized_text_key, entity_type_key), entity_uri_val_emb in unique_entities_map.items():
            if normalized_text_key in text_to_embedding_map and text_to_embedding_map[normalized_text_key]:
                entity_embeddings[entity_uri_val_emb] = text_to_embedding_map[normalized_text_key]

        if entity_embeddings:
            logger.debug(f"\nSuccessfully generated and mapped embeddings for {len(entity_embeddings)} entity URIs.")
            first_uri_with_embedding = next(iter(entity_embeddings.keys()), None)
            if first_uri_with_embedding:
                emb_example = entity_embeddings[first_uri_with_embedding]
                label_for_uri = kg.value(subject=first_uri_with_embedding, predicate=RDFS.label, default=str(first_uri_with_embedding))
                logger.debug(f"  Example embedding for URI <{first_uri_with_embedding}> (Label: '{label_for_uri}'):")
                logger.debug(f"    Vector (first 5 dims): {str(emb_example[:5])}...")
                logger.debug(f"    Vector dimension: {len(emb_example)}")
        else:
            logger.debug("No embeddings were successfully mapped to entity URIs.")
    else:
        logger.debug("No unique entity texts found to generate embeddings for.")
else:
    logger.debug("Skipping embedding generation: No unique entities identified, or LLM client not available.")

if 'entity_embeddings' not in globals():
    entity_embeddings = {}
    logger.debug("Initialized 'entity_embeddings' as an empty dictionary.")

"""
**Output Explanation:**
This block will show:
*   Progress of fetching embeddings.
*   The number of unique entity texts for which embeddings are requested.
*   The number of entity URIs for which embeddings were successfully generated and mapped.
*   An example of an embedding vector (first few dimensions and total length) for one of the entities, along with its URI and label.

### Step 4.2: Link Prediction (Knowledge Discovery - Conceptual) - Cosine Similarity Function
**Task:** Use embeddings to infer new or missing connections.

**Book Concept:** (Ch. 6 - Link prediction as reasoning)
Link prediction aims to identify missing edges (triples) in a KG. KGE models are trained to score potential triples (s, p, o), and high-scoring triples not already in the KG are candidate new links. 

**Methodology (Simplified):**
A full link prediction model is complex. Here, we'll demonstrate a simpler, related concept: **semantic similarity** between entities based on their name embeddings. The `get_cosine_similarity` function calculates the cosine similarity between two embedding vectors. High cosine similarity (close to 1) between entity name embeddings suggests that the entities are semantically related in terms of their textual description. This *could* hint at potential relationships (e.g., two similarly named software products might be competitors or complementary), but it's not direct link prediction for specific predicates.
"""
logger.info("### Step 4.2: Link Prediction (Knowledge Discovery - Conceptual) - Cosine Similarity Function")

def get_cosine_similarity(embedding1, embedding2):
    """Calculates cosine similarity between two embedding vectors using sklearn."""
    if not isinstance(embedding1, (list, np.ndarray)) or not isinstance(embedding2, (list, np.ndarray)):
        return 0.0
    if not embedding1 or not embedding2:
        return 0.0

    vec1 = np.array(embedding1).reshape(1, -1)
    vec2 = np.array(embedding2).reshape(1, -1)

    if vec1.shape[1] != vec2.shape[1]:
        logger.debug(f"Warning: Embedding dimensions do not match for cosine similarity: {vec1.shape[1]} vs {vec2.shape[1]}")
        return 0.0 # Or handle as an error

    return cosine_similarity(vec1, vec2)[0][0]

logger.debug("Function 'get_cosine_similarity' defined.")

"""
**Output Explanation:**
Confirms the definition of the `get_cosine_similarity` function.

#### Link Prediction (Conceptual) - Similarity Calculation Example

**Theory:**
This block demonstrates the use of `get_cosine_similarity`. It selects a couple of entities (preferably organizations, if available with embeddings) from our `entity_embeddings` map and calculates the similarity between their name embeddings. A high similarity might suggest they operate in similar domains or have related roles, which could be a starting point for investigating potential (unobserved) connections if we had a more sophisticated link prediction model.
"""
logger.info("#### Link Prediction (Conceptual) - Similarity Calculation Example")

if len(entity_embeddings) >= 2:
    logger.debug("\nConceptual Link Prediction: Calculating semantic similarity between a sample of entities using their name embeddings.")

    uris_with_embeddings = [uri for uri, emb in entity_embeddings.items() if emb] # Check if emb is not empty

    org_entity_uris_with_embeddings = []
    for uri_cand in uris_with_embeddings:
        rdf_types_for_uri = list(kg.objects(subject=uri_cand, predicate=RDF.type))
        if SCHEMA.Organization in rdf_types_for_uri or EX.ORG in rdf_types_for_uri:
            org_entity_uris_with_embeddings.append(uri_cand)

    entity1_uri_sim = None
    entity2_uri_sim = None

    if len(org_entity_uris_with_embeddings) >= 2:
        entity1_uri_sim = org_entity_uris_with_embeddings[0]
        entity2_uri_sim = org_entity_uris_with_embeddings[1]
        logger.debug(f"Found at least two ORG entities with embeddings for similarity comparison.")
    elif len(uris_with_embeddings) >= 2: # Fallback to any two entities if not enough ORGs
        entity1_uri_sim = uris_with_embeddings[0]
        entity2_uri_sim = uris_with_embeddings[1]
        logger.debug(f"Could not find two ORGs with embeddings. Using two generic entities for similarity comparison.")
    else:
        logger.debug("Not enough entities (less than 2) with valid embeddings to demonstrate similarity.")

    if entity1_uri_sim and entity2_uri_sim:
        embedding1_val = entity_embeddings.get(entity1_uri_sim)
        embedding2_val = entity_embeddings.get(entity2_uri_sim)

        label1_val = kg.value(subject=entity1_uri_sim, predicate=RDFS.label, default=str(entity1_uri_sim))
        label2_val = kg.value(subject=entity2_uri_sim, predicate=RDFS.label, default=str(entity2_uri_sim))

        calculated_similarity = get_cosine_similarity(embedding1_val, embedding2_val)
        logger.debug(f"\n  Similarity between '{label1_val}' (<{entity1_uri_sim}>) and '{label2_val}' (<{entity2_uri_sim}>): {calculated_similarity:.4f}")

        if calculated_similarity > 0.8:
            logger.debug(f"  Interpretation: These entities are highly similar based on their name embeddings.")
        elif calculated_similarity > 0.6:
            logger.debug(f"  Interpretation: These entities show moderate similarity based on their name embeddings.")
        else:
            logger.debug(f"  Interpretation: These entities show low similarity based on their name embeddings.")

    logger.debug("\nNote: This is a conceptual demonstration of semantic similarity. True link prediction involves training specialized KGE models (e.g., TransE, ComplEx) on existing graph triples to predict missing (subject, predicate, object) facts with specific predicates, not just general entity similarity.")
else:
    logger.debug("Skipping conceptual link prediction: Not enough entity embeddings available (need at least 2).")

"""
**Output Explanation:**
This block will:
*   Select two entities that have embeddings.
    *   Calculate and print the cosine similarity score between their embeddings.
    *   Provide a simple interpretation of the similarity score.
    *   Include a disclaimer that this is a simplified concept, not full link prediction.

### Step 4.3: Add Predicted Links (Optional & Conceptual) - Function Definition
**Task:** Integrate high-confidence predicted links into the main graph.

**Book Concept:** (Ch. 6 - KG enrichment and lifecycle)
If a link prediction model were to identify new, high-confidence relationships, these could be added to the KG, enriching it with inferred knowledge. This is part of the KG lifecycle, where the graph evolves and grows.

**Methodology:**
The function `add_inferred_triples_to_graph` is a placeholder to illustrate this. It would take a list of (subject_uri, predicate_uri, object_uri) triples (presumably from a link prediction model) and add them to our main `rdflib.Graph`.
"""
logger.info("### Step 4.3: Add Predicted Links (Optional & Conceptual) - Function Definition")

def add_inferred_triples_to_graph(target_graph, list_of_inferred_triples):
    """Adds a list of inferred (subject_uri, predicate_uri, object_uri) triples to the graph."""
    if not list_of_inferred_triples:
        logger.debug("No inferred triples provided to add.")
        return target_graph, 0

    added_count = 0
    for s_uri, p_uri, o_uri in list_of_inferred_triples:
        if isinstance(s_uri, URIRef) and isinstance(p_uri, URIRef) and (isinstance(o_uri, URIRef) or isinstance(o_uri, Literal)):
            target_graph.add((s_uri, p_uri, o_uri))
            added_count +=1
        else:
            logger.debug(f"  Warning: Skipping malformed conceptual inferred triple: ({s_uri}, {p_uri}, {o_uri})")

    logger.debug(f"Added {added_count} conceptually inferred triples to the graph.")
    return target_graph, added_count

logger.debug("Function 'add_inferred_triples_to_graph' defined.")

"""
**Output Explanation:**
Confirms the definition of the `add_inferred_triples_to_graph` function.

#### Add Predicted Links (Conceptual) - Execution Example

**Theory:**
This block provides a conceptual example. Since we haven't trained a full link prediction model, we create a dummy `conceptual_inferred_triples` list. If this list contained actual high-confidence predictions (e.g., from a TransE model scoring `(CompanyX, ex:potentialAcquirerOf, CompanyY)` highly), the `add_inferred_triples_to_graph` function would integrate them into our `kg`. In this demo, it will likely state that no triples were added unless you manually populate the dummy list.
"""
logger.info("#### Add Predicted Links (Conceptual) - Execution Example")

conceptual_inferred_triples_list = []

SIMILARITY_THRESHOLD_FOR_INFERENCE = 0.85 # Example threshold
if 'entity1_uri_sim' in locals() and 'entity2_uri_sim' in locals() and 'calculated_similarity' in locals():
    if entity1_uri_sim and entity2_uri_sim and calculated_similarity > SIMILARITY_THRESHOLD_FOR_INFERENCE:
        logger.debug(f"Conceptual inference: Entities '{kg.label(entity1_uri_sim)}' and '{kg.label(entity2_uri_sim)}' are highly similar ({calculated_similarity:.2f}).")
        EX.isHighlySimilarTo = EX["isHighlySimilarTo"] # Define if not already
        conceptual_inferred_triples_list.append((entity1_uri_sim, EX.isHighlySimilarTo, entity2_uri_sim))

if conceptual_inferred_triples_list:
    logger.debug(f"\nAttempting to add {len(conceptual_inferred_triples_list)} conceptual inferred triples...")
    kg, num_added = add_inferred_triples_to_graph(kg, conceptual_inferred_triples_list)
    if num_added > 0:
        logger.debug(f"Total triples in graph after adding conceptual inferences: {len(kg)}")
else:
    logger.debug("\nNo conceptual inferred triples generated to add in this demonstration.")

"""
**Output Explanation:**
This block will indicate if any conceptual inferred triples were added to the graph. If the dummy example for high similarity was triggered, it will show that these triples were added and the new total triple count.

## Phase 5: Persistence and Utilization
**(Ref: Ch. 3 – Data Storage; Ch. 5 – Querying and Access)**

**Theory (Phase Overview):**
Once the Knowledge Graph is constructed (and potentially refined), it needs to be stored for long-term access and utilized to derive insights. This phase covers:
*   **KG Storage:** Persisting the graph. Options include RDF serialization formats (like Turtle, RDF/XML), native triple stores (e.g., Fuseki, GraphDB), or graph databases (e.g., Neo4j, if modeled appropriately) (Ch. 3).
*   **Querying and Analysis:** Using query languages like SPARQL (for RDF KGs) to retrieve specific information, answer complex questions, and perform analytical tasks (Ch. 5).
*   **Visualization:** Presenting parts of the KG or query results graphically for better human interpretation and understanding (Ch. 1 & 3 - Value of Visualization in Big Data).

### Step 5.1: Knowledge Graph Storage - Save Function Definition
**Task:** Persist the KG in a suitable format (e.g., RDF/Turtle).

**Book Concept:** (Ch. 3 - Data Storage options)
Serializing the KG to a file allows for persistence, sharing, and loading into other RDF-compliant tools or triple stores.

**Methodology:**
The `save_graph_to_turtle` function uses `rdflib.Graph.serialize()` method to save the `kg` object into a file. Turtle (`.ttl`) is chosen as it's a human-readable and common RDF serialization format.
"""
logger.info("## Phase 5: Persistence and Utilization")

def save_graph_to_turtle(graph_to_save, output_filepath="knowledge_graph.ttl"):
    """Saves the RDF graph to a Turtle file."""
    if not len(graph_to_save):
        logger.debug("Graph is empty. Nothing to save.")
        return False
    try:
        graph_to_save.serialize(destination=output_filepath, format='turtle')
        logger.debug(f"Knowledge Graph with {len(graph_to_save)} triples successfully saved to: {output_filepath}")
        return True
    except Exception as e:
        logger.debug(f"Error saving graph to {output_filepath}: {e}")
        return False

logger.debug("Function 'save_graph_to_turtle' defined.")

"""
**Output Explanation:**
Confirms the definition of the `save_graph_to_turtle` function.

#### Knowledge Graph Storage - Execution

**Theory:**
This block calls the `save_graph_to_turtle` function to persist our constructed `kg` to a file named `tech_acquisitions_kg.ttl`. If the graph contains triples, it will be saved; otherwise, a message indicating an empty graph will be shown.
"""
logger.info("#### Knowledge Graph Storage - Execution")

KG_OUTPUT_FILENAME = "tech_acquisitions_kg.ttl"
if len(kg) > 0:
    logger.debug(f"Attempting to save the graph with {len(kg)} triples...")
    save_graph_to_turtle(kg, KG_OUTPUT_FILENAME)
else:
    logger.debug(f"Knowledge Graph ('kg') is empty. Skipping save to '{KG_OUTPUT_FILENAME}'.")

"""
**Output Explanation:**
This block will print a confirmation message if the graph is successfully saved, including the file path and the number of triples saved. If the graph is empty, it will state that.

### Step 5.2: Querying and Analysis - SPARQL Execution Function
**Task:** Execute SPARQL queries to extract insights.

**Book Concept:** (Ch. 5 - Querying and Access, SPARQL)
SPARQL (SPARQL Protocol and RDF Query Language) is the standard query language for RDF Knowledge Graphs. It allows for pattern matching against the graph structure to retrieve data, infer new information (through more complex queries), and answer analytical questions.

**Methodology:**
The `execute_sparql_query` function takes our `rdflib.Graph` and a SPARQL query string. It uses `graph.query()` to execute the query and then iterates through the results, printing them in a readable format. This function will be used to run several example queries.
"""
logger.info("### Step 5.2: Querying and Analysis - SPARQL Execution Function")

def execute_sparql_query(graph_to_query, query_string_sparql):
    """Executes a SPARQL query on the graph and prints results, returning them as a list of dicts."""
    if not len(graph_to_query):
        logger.debug("Cannot execute SPARQL query: The graph is empty.")
        return []

    logger.debug(f"\nExecuting SPARQL Query:\n{query_string_sparql}")
    try:
        query_results = graph_to_query.query(query_string_sparql)
    except Exception as e:
        logger.debug(f"Error executing SPARQL query: {e}")
        return []

    if not query_results:
        logger.debug("Query executed successfully but returned no results.")
        return []

    results_list_of_dicts = []
    logger.debug(f"Query Results ({len(query_results)} found): ")
    for row_idx, row_data in enumerate(query_results):
        result_item_dict = {}
        if hasattr(row_data, 'labels'): # rdflib 6.x+ provides .labels and .asdict()
            result_item_dict = {str(label): str(value) for label, value in row_data.asdict().items()}
        else: # Fallback for older rdflib or if .asdict() is not available
            result_item_dict = {f"col_{j}": str(item_val) for j, item_val in enumerate(row_data)}

        results_list_of_dicts.append(result_item_dict)

        if row_idx < 10: # Print up to 10 results
            logger.debug(f"  Row {row_idx+1}: {result_item_dict}")
        elif row_idx == 10:
            logger.debug(f"  ... ( dalších {len(query_results) - 10} výsledků )") # and more results in Czech, should be English
            logger.debug(f"  ... (and {len(query_results) - 10} more results)")

    return results_list_of_dicts

logger.debug("Function 'execute_sparql_query' defined.")

"""
**Output Explanation:**
Confirms the definition of the `execute_sparql_query` function.

#### SPARQL Querying and Analysis - Execution Examples

**Theory:**
This block demonstrates the power of SPARQL by executing several example queries against our constructed KG (`kg`). Each query targets different aspects of the acquisition data:
*   **Query 1: List Organizations:** Retrieves all entities explicitly typed as `schema:Organization` (or `ex:ORG`) and their labels. This is a basic check to see what organizations are in our KG.
*   **Query 2: Find Acquisition Relationships:** Identifies pairs of companies where one acquired the other, based on our `ex:ACQUIRED` predicate. This directly extracts acquisition events.
*   **Query 3: Find Acquisitions with Price:** Retrieves companies (or acquisition events) that have an associated `ex:HAS_PRICE` relationship pointing to a monetary value (`schema:PriceSpecification` or `ex:MONEY`).
The results of each query are printed, showcasing how structured queries can extract specific insights from the graph.
"""
logger.info("#### SPARQL Querying and Analysis - Execution Examples")

if len(kg) > 0:
    logger.debug("\n--- Executing Sample SPARQL Queries ---")
    sparql_query_1 = """
    PREFIX ex: <http://example.org/kg/>
    PREFIX schema: <http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?org_uri ?org_label
    WHERE {
        ?org_uri a schema:Organization ;
                 rdfs:label ?org_label .
    }
    ORDER BY ?org_label
    LIMIT 10
    """
    query1_results = execute_sparql_query(kg, sparql_query_1)

    sparql_query_2 = """
    PREFIX ex: <http://example.org/kg/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>

    SELECT ?acquiredCompanyLabel ?acquiringCompanyLabel
    WHERE {
        ?acquiredCompany ex:ACQUIRED ?acquiringCompany .
        ?acquiredCompany rdfs:label ?acquiredCompanyLabel .
        ?acquiringCompany rdfs:label ?acquiringCompanyLabel .
        ?acquiredCompany a schema:Organization .
        ?acquiringCompany a schema:Organization .
    }
    LIMIT 10
    """
    query2_results = execute_sparql_query(kg, sparql_query_2)

    sparql_query_3 = """
    PREFIX ex: <http://example.org/kg/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>

    SELECT ?companyLabel ?priceLabel ?dateLabel
    WHERE {
        ?company ex:HAS_PRICE ?priceEntity .
        ?company rdfs:label ?companyLabel .
        ?priceEntity rdfs:label ?priceLabel .
        ?company a schema:Organization .
        ?priceEntity a schema:PriceSpecification .

        OPTIONAL {
            ?company ex:ANNOUNCED_ON ?dateEntity .
            ?dateEntity rdfs:label ?dateLabelRaw .
            BIND(COALESCE(?dateLabelRaw, STR(?dateEntity)) As ?dateLabel)
        }
    }
    LIMIT 10
    """
    query3_results = execute_sparql_query(kg, sparql_query_3)

    sparql_query_4 = """
    PREFIX ex: <http://example.org/kg/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>

    SELECT ?acquiringCompanyLabel (COUNT(?acquiredCompany) AS ?numberOfAcquisitions)
    WHERE {
        ?acquiredCompany ex:ACQUIRED ?acquiringCompany .
        ?acquiringCompany rdfs:label ?acquiringCompanyLabel .
        ?acquiringCompany a schema:Organization .
        ?acquiredCompany a schema:Organization .
    }
    GROUP BY ?acquiringCompanyLabel
    ORDER BY DESC(?numberOfAcquisitions)
    LIMIT 10
    """
    query4_results = execute_sparql_query(kg, sparql_query_4)

else:
    logger.debug("Knowledge Graph ('kg') is empty. Skipping SPARQL query execution.")

"""
**Output Explanation:**
This block will print:
*   Each SPARQL query string.
*   The results (up to a limit) for each query, typically as a list of dictionaries where keys are the `SELECT` variables.
If the KG is empty, it will indicate that queries are skipped.

### Step 5.3: Visualization (Optional) - Visualization Function Definition
**Task:** Visualize parts of the KG or results from queries for better interpretability.

**Book Concept:** (Ch. 1 & 3 - Visualization in Big Data)
Visualizing graph structures can make complex relationships much easier to understand for humans. Interactive visualizations allow for exploration and discovery.

**Methodology:**
The `visualize_subgraph_pyvis` function uses the `pyvis` library to create an interactive HTML-based network visualization. It:
*   Takes the `rdflib.Graph` and an optional filename.
    *   (A more advanced version could take a central node URI and depth to explore from that node).
*   For simplicity in this demo, it visualizes a sample of triples from the graph.
*   Adds nodes and edges to a `pyvis.Network` object.
*   Nodes are labeled with their `rdfs:label` (or a part of their URI if no label).
*   Edges are labeled with the predicate name.
*   Saves the visualization to an HTML file and attempts to display it inline if in a Jupyter environment.
"""
logger.info("### Step 5.3: Visualization (Optional) - Visualization Function Definition")

def visualize_subgraph_pyvis(graph_to_viz, output_filename="kg_visualization.html", sample_size_triples=75):
    """Visualizes a sample subgraph using pyvis and saves to HTML."""
    if not len(graph_to_viz):
        logger.debug("Graph is empty, nothing to visualize.")
        return None

    net = Network(notebook=True, height="800px", width="100%", cdn_resources='remote', directed=True)
    net.repulsion(node_distance=150, spring_length=200)
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "timestep": 0.5,
        "stabilization": {"iterations": 150}
      }
    }
    """)

    added_nodes_set = set()

    triples_for_visualization = list(graph_to_viz)[:min(sample_size_triples, len(graph_to_viz))]

    if not triples_for_visualization:
        logger.debug("No triples selected from the sample for visualization.")
        return None

    logger.debug(f"Preparing visualization for {len(triples_for_visualization)} sample triples...")

    for s_uri, p_uri, o_val in tqdm(triples_for_visualization, desc="Building Pyvis Visualization"):
        s_label_str = str(graph_to_viz.value(subject=s_uri, predicate=RDFS.label, default=s_uri.split('/')[-1].split('#')[-1]))
        p_label_str = str(p_uri.split('/')[-1].split('#')[-1])

        s_node_id = str(s_uri)
        s_node_title = f"{s_label_str}\nURI: {s_uri}"
        s_node_group_uri = graph_to_viz.value(s_uri, RDF.type)
        s_node_group = str(s_node_group_uri.split('/')[-1].split('#')[-1]) if s_node_group_uri else "UnknownType"


        if s_uri not in added_nodes_set:
            net.add_node(s_node_id, label=s_label_str, title=s_node_title, group=s_node_group)
            added_nodes_set.add(s_uri)

        if isinstance(o_val, URIRef): # If object is a resource, add it as a node and draw an edge
            o_label_str = str(graph_to_viz.value(subject=o_val, predicate=RDFS.label, default=o_val.split('/')[-1].split('#')[-1]))
            o_node_id = str(o_val)
            o_node_title = f"{o_label_str}\nURI: {o_val}"
            o_node_group_uri = graph_to_viz.value(o_val, RDF.type)
            o_node_group = str(o_node_group_uri.split('/')[-1].split('#')[-1]) if o_node_group_uri else "UnknownType"

            if o_val not in added_nodes_set:
                net.add_node(o_node_id, label=o_label_str, title=o_node_title, group=o_node_group)
                added_nodes_set.add(o_val)
            net.add_edge(s_node_id, o_node_id, title=p_label_str, label=p_label_str)
        else: # If object is a literal, add it as a property to the subject node's title (tooltip)
            for node_obj in net.nodes:
                if node_obj['id'] == s_node_id:
                    node_obj['title'] += f"\n{p_label_str}: {str(o_val)}"
                    break

    try:
        net.save_graph(output_filename)
        logger.debug(f"Interactive KG visualization saved to HTML file: {output_filename}")
    except Exception as e:
        logger.debug(f"Error saving or attempting to show graph visualization: {e}")
    return net # Return the network object

logger.debug("Function 'visualize_subgraph_pyvis' defined.")

"""
**Output Explanation:**
Confirms the definition of the `visualize_subgraph_pyvis` function.

#### KG Visualization - Execution

**Theory:**
This block calls `visualize_subgraph_pyvis` with our `kg`. It will generate an HTML file (e.g., `tech_acquisitions_kg_viz.html`) containing the interactive graph. If running in a compatible Jupyter environment, the visualization might also render directly in the notebook output. This allows for a visual exploration of the connections and entities within a sample of our KG.
"""
logger.info("#### KG Visualization - Execution")

VIZ_OUTPUT_FILENAME = "tech_acquisitions_kg_interactive_viz.html"
pyvis_network_object = None # Initialize

if len(kg) > 0:
    logger.debug(f"Attempting to visualize a sample of the graph with {len(kg)} triples...")
    pyvis_network_object = visualize_subgraph_pyvis(kg, output_filename=VIZ_OUTPUT_FILENAME, sample_size_triples=75)
else:
    logger.debug(f"Knowledge Graph ('kg') is empty. Skipping visualization.")

if pyvis_network_object is not None:
    try:
        logger.debug(f"\nTo view the visualization, open the file '{VIZ_OUTPUT_FILENAME}' in a web browser.")
        logger.debug("If in a Jupyter Notebook/Lab, the graph might also be rendered above this message.")
    except Exception as e_display:
        logger.debug(f"Could not automatically display visualization inline ({e_display}). Please open '{VIZ_OUTPUT_FILENAME}' manually.")

if pyvis_network_object:
    pyvis_network_object # This line is crucial for auto-display in some Jupyter environments

"""
**Output Explanation:**
This block will:
*   Generate an HTML file (e.g., `tech_acquisitions_kg_interactive_viz.html`) with the interactive graph visualization.
*   Print a message confirming the save and provide the filename.
*   If in a compatible Jupyter environment, it might also render the graph directly below the cell. Otherwise, you'll need to open the HTML file manually in a browser.

## Conclusion and Future Work

This notebook has demonstrated a comprehensive, albeit simplified, end-to-end pipeline for constructing a Knowledge Graph from unstructured news articles, with a focus on technology company acquisitions. We navigated through critical phases, referencing conceptual underpinnings from Big Data and Knowledge Graph literature:

1.  **Data Acquisition and Preparation:** We loaded articles from the CNN/DailyMail dataset and performed essential cleaning to prepare the text for analysis. This underscored the importance of data quality as a foundation (Ch. 1, Ch. 3).
2.  **Information Extraction:** 
    *   Named Entity Recognition (NER) was performed first exploratively with spaCy, then more targetedly using an LLM guided by a refined entity schema. This created the *nodes* of our graph (Ch. 2).
    *   Relationship Extraction (RE) using an LLM identified semantic connections between these entities, forming the *edges* (Ch. 2).
3.  **Knowledge Graph Construction:** 
    *   Entities were normalized for consistency, and unique URIs were generated, aiding in entity resolution (Ch. 6, Ch. 8).
    *   Extracted information was mapped to a schema (mixing custom `EX:` terms and `schema.org` terms) and materialized into RDF triples using `rdflib` (Ch. 2, Ch. 4).
4.  **Knowledge Graph Refinement (Conceptual):** 
    *   We generated text embeddings for entity names, bridging symbolic and sub-symbolic representations (Ch. 6).
    *   The concept of link prediction via semantic similarity was introduced, hinting at KG enrichment capabilities (Ch. 6).
5.  **Persistence and Utilization:** 
    *   The KG was persisted by serializing it to a Turtle file (Ch. 3).
    *   SPARQL queries were executed to retrieve structured insights, demonstrating the analytical power of KGs (Ch. 5).
    *   A sample subgraph was visualized, highlighting the importance of making KGs accessible (Ch. 1, Ch. 3).

### Potential Future Enhancements:
*   **Advanced Entity Disambiguation & Linking (EDL):** Implement robust EDL to link extracted entities to canonical entries in external KGs like Wikidata or DBpedia. This would greatly improve graph integration and consistency.
*   **Richer Ontology/Schema:** Develop a more detailed custom ontology for technology acquisitions or align more comprehensively with existing financial or business ontologies (e.g., FIBO).
*   **Sophisticated Relationship Extraction:** Explore more advanced RE techniques, including classifying a wider range of relation types, handling n-ary relations, and event extraction (modeling acquisitions as complex events with multiple participants and attributes).
*   **Knowledge Graph Embedding Models:** Train dedicated KGE models (e.g., TransE, ComplEx, RotatE from Ch. 6) on the generated triples for more accurate link prediction and KG completion.
*   **Reasoning and Inference:** Implement ontological reasoning (e.g., using RDFS/OWL reasoners) to infer new facts based on the schema and asserted triples.
*   **Scalability and Performance:** For larger datasets, utilize distributed processing frameworks (like Apache Spark, conceptually linked to SANSA in Ch. 7 for ML on KGs) and deploy the KG in a scalable graph database or triple store.
*   **LLM Fine-tuning:** Fine-tune smaller, open-source LLMs specifically on NER and RE tasks for the technology/financial domain to potentially achieve better performance and cost-efficiency than general-purpose models for these specific tasks.
*   **Temporal Dynamics:** Incorporate the temporal aspect of news data more explicitly, tracking how information and relationships evolve over time.
*   **User Interface:** Develop a user-friendly interface for exploring, querying, and visualizing the KG beyond programmatic access.

This project serves as a foundational example, illustrating how modern NLP techniques, particularly LLMs, can be integrated with traditional KG methodologies to extract and structure valuable knowledge from vast amounts of unstructured text.
"""
logger.info("## Conclusion and Future Work")

logger.info("\n\n[DONE]", bright=True)