from jet.file.utils import load_file
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from jet.transformers.formatters import format_json
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    SchemaLLMPathExtractor,
    DynamicLLMPathExtractor,
)
from jet.llm.ollama.base import Ollama
from llama_index.core import Settings
import wikipedia
import os

initialize_ollama_settings()

"""
# Comparing LLM Path Extractors for Knowledge Graph Construction

In this notebook, we'll compare three different LLM Path Extractors from llama_index:
1. SimpleLLMPathExtractor
2. SchemaLLMPathExtractor
3. DynamicLLMPathExtractor (New)

We'll use a Wikipedia page as our test data and visualize the resulting knowledge graphs using Pyvis.

## Setup and Imports
"""

# !pip install llama_index pyvis wikipedia


# import nest_asyncio

# nest_asyncio.apply()

"""
## Set up LLM Backend
"""

# os.environ["OPENAI_API_KEY"] = "sk-proj-..."
model = "llama3.2"
llm = Ollama(temperature=0.0, model=model,
             request_timeout=300.0, context_window=OLLAMA_MODEL_EMBEDDING_TOKENS[model])

Settings.llm = llm
Settings.chunk_size = 2048
Settings.chunk_overlap = 20

"""
## Fetch Some Raw Text from Wikipedia
"""


def get_wikipedia_content(title):
    try:
        page = wikipedia.page(title)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        logger.debug(f"Disambiguation page. Options: {e.options}")
    except wikipedia.exceptions.PageError:
        logger.debug(f"Page '{title}' does not exist.")
    return None


# wiki_title = "Barack Obama"
# content = get_wikipedia_content(wiki_title)

data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
data: dict[str, list[str]] = load_file(data_file)
texts = [text for texts in list(data.values()) for text in texts]
title = "I'll Become a Villainess Who Goes Down in History"
documents = [Document(text=text, metadata={"title": title}) for text in texts]

# if content:
#     document = Document(text=content, metadata={"title": wiki_title})
#     logger.debug(
#         f"Fetched content for '{wiki_title}' (length: {len(content)} characters)"
#     )
# else:
#     logger.debug("Failed to fetch Wikipedia content.")

"""
## 1. SimpleLLMPathExtractor
"""

kg_extractor = SimpleLLMPathExtractor(
    llm=llm, max_paths_per_chunk=3, num_workers=4
)

simple_index = PropertyGraphIndex.from_documents(
    # [document],
    documents,
    llm=llm,
    embed_kg_nodes=False,
    kg_extractors=[kg_extractor],
    show_progress=True,
)

simple_index.property_graph_store.save_networkx_graph(
    name="./generated/SimpleGraph.html"
)

logger.debug("Simple property graph store triplets:")
logger.orange(format_json(
    simple_index.property_graph_store.get_triplets(
        entity_names=["Anime", "Episode"]
    )[:5]
))


"""
## 2. DynamicLLMPathExtractor

### Without intial ontology :
Here, we let the LLM define the ontology on the fly, giving it full freedom to label the nodes as it best sees fit.
"""

kg_extractor = DynamicLLMPathExtractor(
    llm=llm,
    max_triplets_per_chunk=10,
    num_workers=4,
    allowed_entity_types=None,
    allowed_relation_types=None,
    allowed_relation_props=[],
    allowed_entity_props=[],

)

dynamic_index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_kg_nodes=False,
    kg_extractors=[kg_extractor],
    show_progress=True,
)

dynamic_index.property_graph_store.save_networkx_graph(
    name="./generated/DynamicGraph.html"
)


logger.debug("Dynamic property graph store triplets:")
logger.orange(format_json(
    dynamic_index.property_graph_store.get_triplets(
        entity_names=["Anime", "Episode"]
    )[:5]
))


"""
### With initial ontology for guided KG extraction : 
Here, we have partial knowledge of what we want to detect, we know the article is about Barack Obama, so we define some entities and relations that could help guide the LLM in the labeling process as it detects the entities and relations. This doesn't guarantee that the LLM will use them, it simply guides it and gives it some ideas. It will still be up to the LLM to decide whether it uses the entities and relations we provide or not.
"""

kg_extractor = DynamicLLMPathExtractor(
    llm=llm,
    max_triplets_per_chunk=10,
    num_workers=4,
    allowed_entity_types=["Episode"],
    allowed_relation_types=["HAS_EPISODE",
                            "PART_OF_SEASON", "HAS_GENRE", "HAS_TITLE"],
    allowed_relation_props=["episodes"],
    allowed_entity_props=["title", "seasons", "episodes",
                          "synopsis", "release_date", "end_date", "status"],
)

dynamic_index_2 = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_kg_nodes=False,
    kg_extractors=[kg_extractor],
    show_progress=True,
)

dynamic_index_2.property_graph_store.save_networkx_graph(
    name="./generated/DynamicGraph.html"
)

logger.debug("Dynamic property graph store 2 triplets:")
logger.orange(format_json(
    dynamic_index_2.property_graph_store.get_triplets(
        entity_names=["Anime", "Episode"]
    )[:5]
))


"""
# 3 - SchemaLLMPathExtractor
"""

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    max_triplets_per_chunk=10,
    strict=False,  # Set to False to showcase why it's not going to be the same as DynamicLLMPathExtractor
    # USE DEFAULT ENTITIES (PERSON, ORGANIZATION... etc)
    possible_entities=None,
    possible_relations=None,  # USE DEFAULT RELATIONSHIPS
    possible_relation_props=[
        "extra_description"
    ],  # Set to `None` to skip property generation
    possible_entity_props=[
        "extra_description"
    ],  # Set to `None` to skip property generation
    num_workers=4,
)

schema_index = PropertyGraphIndex.from_documents(
    # [document],
    documents,
    llm=llm,
    embed_kg_nodes=False,
    kg_extractors=[kg_extractor],
    show_progress=True,
)

schema_index.property_graph_store.save_networkx_graph(
    name="./generated/SchemaGraph.html"
)

logger.debug("Schema property graph store triplets:")
logger.orange(format_json(
    schema_index.property_graph_store.get_triplets(
        entity_names=["Anime", "Episode"]
    )[:5]
))

"""
## Comparison and Analysis

Let's compare the results of the three extractors:

1. **SimpleLLMPathExtractor**: This extractor creates a basic knowledge graph without any predefined schema. It may produce a larger number of diverse relationships but might lack consistency in entity and relation naming.


3. **DynamicLLMPathExtractor**: 
    - This new extractor combines the flexibility of SimpleLLMPathExtractor with some initial guidance from a schema. It can expand beyond the initial entity and relation types, potentially producing a rich and diverse graph while maintaining some level of consistency. 
    - Not giving it any entities or relations to start with in the input gives the LLM complete freedom to infer the schema on the fly as it best sees fit. This is going to vary based on the LLM and the temperature used.

3. **SchemaLLMPathExtractor**: With a predefined schema, this extractor produces a more structured graph. The entities and relations are limited to those specified in the schema, which can lead to a more consistent but potentially less comprehensive graph. Even if we set "strict" to false, the extracted KG Graph doesn't reflect the LLM's pursuit of trying to find new entities and types that fall outside of the input schema's scope.


## Key observations:

- The SimpleLLMPathExtractor graph might have the most diverse set of entities and relations.
- The SchemaLLMPathExtractor graph should be the most consistent but might miss a lot of relationships that don't fit the predefined schema, even if we don't impose a strict validation of the schema.
- The DynamicLLMPathExtractor graph should show a balance between diversity and consistency, potentially capturing important relationships that the schema-based approach might miss while still maintaining some structure.

## The choice between these extractors depends on the specific use case:

- Use SimpleLLMPathExtractor for exploratory analysis where you want to capture a wide range of potential relationships for RAG applications, without caring about the entity types.
- Use SchemaLLMPathExtractor when you have a well-defined domain and want to ensure consistency in the extracted knowledge.
- Use DynamicLLMPathExtractor when you want a balance between structure and flexibility, allowing the model to discover new entity and relation types while still providing some initial guidance. This one is especially useful if you want a KG with labeled (typed) entities but don't have an input Schema (or you've partially defined the schema as a starting base).
"""

logger.info("\n\n[DONE]", bright=True)
