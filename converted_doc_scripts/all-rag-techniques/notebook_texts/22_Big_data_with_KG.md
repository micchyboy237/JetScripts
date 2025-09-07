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

**Output Explanation:**
This block simply confirms that the necessary libraries have been imported without error.

#### Initialize LLM Client and spaCy Model

**Theory:**
Here, we instantiate the clients for our primary NLP tools:
*   **OpenAI Client:** Configured to point to the Nebius API. This client will be used to send requests to the deployed LLM for tasks like entity extraction, relation extraction, and generating embeddings. A basic check is performed to see if the configuration parameters are set.
*   **spaCy Model:** We load `en_core_web_sm`, a small English model from spaCy. This model provides efficient capabilities for tokenization, part-of-speech tagging, lemmatization, and basic Named Entity Recognition (NER). It's useful for initial text exploration and can complement LLM-based approaches.

**Output Explanation:**
This block prints messages indicating the status of the OpenAI client and spaCy model initialization. Warnings are shown if configurations are missing or models can't be loaded.

#### Define RDF Namespaces

**Theory:**
In RDF, namespaces are used to avoid naming conflicts and to provide context for terms (URIs).
*   `EX`: A custom namespace for terms specific to our project (e.g., our entities and relationships if not mapped to standard ontologies).
*   `SCHEMA`: Refers to Schema.org, a widely used vocabulary for structured data on the internet. We'll try to map some of our extracted types to Schema.org terms for better interoperability.
*   `RDFS`: RDF Schema, provides basic vocabulary for describing RDF vocabularies (e.g., `rdfs:label`, `rdfs:Class`).
*   `RDF`: The core RDF vocabulary (e.g., `rdf:type`).
*   `XSD`: XML Schema Datatypes, used for specifying literal data types (e.g., `xsd:string`, `xsd:date`).
*   `SKOS`: Simple Knowledge Organization System, useful for thesauri, taxonomies, and controlled vocabularies (e.g., `skos:altLabel` for alternative names).

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

**Output Explanation:**
This cell defines the `acquire_articles` function. It will print a confirmation once the function is defined in the Python interpreter's memory.

#### Execute Data Acquisition

**Theory:**
Now we call the `acquire_articles` function. We define keywords relevant to our goal (technology company acquisitions) to guide the filtering process. A `SAMPLE_SIZE` is set to keep the amount of data manageable for this demonstration. Smaller samples allow for faster iteration, especially when using LLMs which can have associated costs and latency.

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

**Output Explanation:**
Confirms that the `clean_article_text` function, which will be used to preprocess article content, has been defined.

#### Execute Data Cleaning

**Theory:**
This block iterates through the `raw_data_sample` (acquired in the previous step). For each article, it calls the `clean_article_text` function. The cleaned text, along with the original article ID and potentially other useful fields like 'summary' (if available from the dataset as 'highlights'), is stored in a new list called `cleaned_articles`. This new list will be the primary input for the subsequent Information Extraction phase.

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

**Output Explanation:**
Confirms the definition of the `get_spacy_entity_counts` function.

#### 2.1.1: Entity Exploration with spaCy - Plotting Function Definition

**Theory:**
The `plot_entity_distribution` function takes the entity counts (from `get_spacy_entity_counts`) and uses `matplotlib` to generate a bar chart. Visualizing this distribution helps in quickly identifying the most frequent entity types, which can inform subsequent decisions about which types to prioritize for the KG.

**Output Explanation:**
Confirms the definition of the `plot_entity_distribution` function.

#### 2.1.1: Entity Exploration with spaCy - Execution

**Theory:**
This block executes the spaCy-based entity exploration. It calls `get_spacy_entity_counts` on the `cleaned_articles`. The resulting counts are then printed and passed to `plot_entity_distribution` to visualize the findings. This step is skipped if no cleaned articles are available or if the spaCy model failed to load.

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

**Output Explanation:**
Confirms the definition of the `normalize_entity_text` function.

#### Execute Entity Normalization and URI Generation

**Theory:**
This block processes the `articles_with_relations` list. For each entity extracted by the LLM:
1.  Its text is normalized using `normalize_entity_text`.
2.  A unique URI (Uniform Resource Identifier) is generated for each distinct normalized entity. We use a simple scheme: `EX:<NormalizedText>_<EntityType>`. The `EX` is our custom namespace. This creates a canonical identifier for each unique real-world concept (as per our normalization). A `unique_entities_map` dictionary stores the mapping from `(normalized_text, entity_type)` to its URI, ensuring that the same normalized entity always gets the same URI across all articles.
3.  The normalized text and the URI are added to the entity's dictionary.
The results are stored in `articles_with_normalized_entities`.

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

**Output Explanation:**
Confirms the definition of the `get_rdf_type_for_entity` mapping function.

#### Schema/Ontology Alignment - RDF Predicate Mapping Function

**Theory:**
The `get_rdf_predicate` function maps our string-based relationship predicates (e.g., "ACQUIRED", "HAS_PRICE" from the LLM RE step) to RDF Properties. For simplicity and custom control, these are typically mapped to properties within our `EX` namespace. The function ensures that predicate strings are converted into valid URI components (e.g., by replacing spaces with underscores).

**Output Explanation:**
Confirms the definition of the `get_rdf_predicate` mapping function.

#### Schema/Ontology Alignment - Examples

**Theory:**
This block simply prints out a few examples of how our entity types and relationship predicates would be mapped to RDF terms using the functions defined above. This serves as a quick check and illustration of the mapping logic.

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
*   Use the configured OpenAI/Nebius embedding API (with `EMBEDDING_MODEL_NAME`) to fetch pre-trained embeddings for these texts.
*   Handle batching or individual requests as appropriate for the API.
*   Return a dictionary mapping each input text to its embedding vector.
These embeddings represent the semantic meaning of the entity names.

**Output Explanation:**
Confirms the definition of the `get_embeddings_for_texts` function.

#### Generate KG Embeddings - Execution

**Theory:**
This block orchestrates the generation of embeddings for our KG entities:
1.  It extracts the set of unique, normalized entity texts from our `unique_entities_map` (which maps `(normalized_text, type)` to URIs). We are interested in embedding the textual representation of entities.
2.  It calls `get_embeddings_for_texts` with this list of unique texts.
3.  The returned embeddings (which are mapped to texts) are then re-mapped to our canonical entity URIs, creating the `entity_embeddings` dictionary: `{entity_uri: embedding_vector}`.
This dictionary will store the vector representation for each unique entity in our graph.

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

**Output Explanation:**
Confirms the definition of the `get_cosine_similarity` function.

#### Link Prediction (Conceptual) - Similarity Calculation Example

**Theory:**
This block demonstrates the use of `get_cosine_similarity`. It selects a couple of entities (preferably organizations, if available with embeddings) from our `entity_embeddings` map and calculates the similarity between their name embeddings. A high similarity might suggest they operate in similar domains or have related roles, which could be a starting point for investigating potential (unobserved) connections if we had a more sophisticated link prediction model.

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

**Output Explanation:**
Confirms the definition of the `add_inferred_triples_to_graph` function.

#### Add Predicted Links (Conceptual) - Execution Example

**Theory:**
This block provides a conceptual example. Since we haven't trained a full link prediction model, we create a dummy `conceptual_inferred_triples` list. If this list contained actual high-confidence predictions (e.g., from a TransE model scoring `(CompanyX, ex:potentialAcquirerOf, CompanyY)` highly), the `add_inferred_triples_to_graph` function would integrate them into our `kg`. In this demo, it will likely state that no triples were added unless you manually populate the dummy list.

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

**Output Explanation:**
Confirms the definition of the `save_graph_to_turtle` function.

#### Knowledge Graph Storage - Execution

**Theory:**
This block calls the `save_graph_to_turtle` function to persist our constructed `kg` to a file named `tech_acquisitions_kg.ttl`. If the graph contains triples, it will be saved; otherwise, a message indicating an empty graph will be shown.

**Output Explanation:**
This block will print a confirmation message if the graph is successfully saved, including the file path and the number of triples saved. If the graph is empty, it will state that.

### Step 5.2: Querying and Analysis - SPARQL Execution Function
**Task:** Execute SPARQL queries to extract insights.

**Book Concept:** (Ch. 5 - Querying and Access, SPARQL)
SPARQL (SPARQL Protocol and RDF Query Language) is the standard query language for RDF Knowledge Graphs. It allows for pattern matching against the graph structure to retrieve data, infer new information (through more complex queries), and answer analytical questions.

**Methodology:**
The `execute_sparql_query` function takes our `rdflib.Graph` and a SPARQL query string. It uses `graph.query()` to execute the query and then iterates through the results, printing them in a readable format. This function will be used to run several example queries.

**Output Explanation:**
Confirms the definition of the `execute_sparql_query` function.

#### SPARQL Querying and Analysis - Execution Examples

**Theory:**
This block demonstrates the power of SPARQL by executing several example queries against our constructed KG (`kg`). Each query targets different aspects of the acquisition data:
*   **Query 1: List Organizations:** Retrieves all entities explicitly typed as `schema:Organization` (or `ex:ORG`) and their labels. This is a basic check to see what organizations are in our KG.
*   **Query 2: Find Acquisition Relationships:** Identifies pairs of companies where one acquired the other, based on our `ex:ACQUIRED` predicate. This directly extracts acquisition events.
*   **Query 3: Find Acquisitions with Price:** Retrieves companies (or acquisition events) that have an associated `ex:HAS_PRICE` relationship pointing to a monetary value (`schema:PriceSpecification` or `ex:MONEY`).
The results of each query are printed, showcasing how structured queries can extract specific insights from the graph.

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

**Output Explanation:**
Confirms the definition of the `visualize_subgraph_pyvis` function.

#### KG Visualization - Execution

**Theory:**
This block calls `visualize_subgraph_pyvis` with our `kg`. It will generate an HTML file (e.g., `tech_acquisitions_kg_viz.html`) containing the interactive graph. If running in a compatible Jupyter environment, the visualization might also render directly in the notebook output. This allows for a visual exploration of the connections and entities within a sample of our KG.

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
