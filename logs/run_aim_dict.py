from aim import Run, Text
from jet.logger import logger


run_settings = {
    "log_system_params": True,
    "repo": "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/aim-logs",
    "experiment": "Code Scraper Test"
}

# Initialize a new run
run = Run(**run_settings)

track_settings = {
    "name": "docs.llamaindex.ai",
    "context": {
        "model": "codellama",
        "options": {
            "seed": 42,
            "num_ctx": 4096,
            "num_keep": 0,
            "num_predict": -1,
            "temperature": 0
        },
        "type": "code_scraper",
        "url_id": "docs_llamaindex_ai_en_stable_examples_node_postprocessor_optimizerdemo",
        "url": "https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/OptimizerDemo/"
    }
}

value = """
## System

```
You are an AI assistant that follows instructions. You can understand and write code of any language, extract code from structured and unstructured content, and provide real-world usage examples. You can write clean, optimized, readable, and modular code. You follow best practices and correct syntax.
```

## Prompt

```
Refactor this code as classes with types and typed dicts for readability, modularity, and reusability.
Add main function for real world usage examples.
Generated code should be complete and working with correct syntax.
Include logs and progress tracking if applicable
Add comments to explain each function and show installation instructions if dependencies are provided.

Output only the Python code wrapped in a code block without additional information (use ```python).

# Sentence Embedding Optimizer
This postprocessor optimizes token usage by removing sentences that are not relevant to the query (this is done using embeddings).The percentile cutoff is a measure for using the top percentage of relevant sentences. The threshold cutoff can be specified instead, which uses a raw similarity cutoff for picking which sentences to keep.
In [ ]:
Copied!
    %pip install llama-index-readers-wikipedia
%pip install llama-index-readers-wikipedia
In [ ]:
Copied!
    # My OpenAI Key
    import os
    os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
# My OpenAI Key import os os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
### Setup
If you're opening this Notebook on colab, you will probably need to install LlamaIndex .
In [ ]:
Copied!
    !pip install llama-index
!pip install llama-index
In [ ]:
Copied!
    from llama_index.core import download_loader
    from llama_index.readers.wikipedia import WikipediaReader
    loader = WikipediaReader()
    documents = loader.load_data(pages=["Berlin"])
from llama_index.core import download_loader from llama_index.readers.wikipedia import WikipediaReader loader = WikipediaReader() documents = loader.load_data(pages=["Berlin"])
In [ ]:
Copied!
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents)
from llama_index.core import VectorStoreIndex index = VectorStoreIndex.from_documents(documents)
    <class 'llama_index.readers.schema.base.Document'>
    INFO:root:> [build_index_from_documents] Total LLM token usage: 0 tokens
    INFO:root:> [build_index_from_documents] Total embedding token usage: 18390 tokens
Compare query with and without optimization for LLM token usage, Embedding Model usage on query, Embedding model usage for optimizer, and total time.
In [ ]:
Copied!
    import time
    from llama_index.core import VectorStoreIndex
    from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
    print("Without optimization")
    start_time = time.time()
    query_engine = index.as_query_engine()
    res = query_engine.query("What is the population of Berlin?")
    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))
    print("Answer: {}".format(res))
    print("With optimization")
    start_time = time.time()
    query_engine = index.as_query_engine(
        node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)]
    )
    res = query_engine.query("What is the population of Berlin?")
    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))
    print("Answer: {}".format(res))
    print("Alternate optimization cutoff")
    start_time = time.time()
    query_engine = index.as_query_engine(
        node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.7)]
    )
    res = query_engine.query("What is the population of Berlin?")
    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))
    print("Answer: {}".format(res))
import time from llama_index.core import VectorStoreIndex from llama_index.core.postprocessor import SentenceEmbeddingOptimizer print("Without optimization") start_time = time.time() query_engine = index.as_query_engine() res = query_engine.query("What is the population of Berlin?") end_time = time.time() print("Total time elapsed: {}".format(end_time - start_time)) print("Answer: {}".format(res)) print("With optimization") start_time = time.time() query_engine = index.as_query_engine( node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)] ) res = query_engine.query("What is the population of Berlin?") end_time = time.time() print("Total time elapsed: {}".format(end_time - start_time)) print("Answer: {}".format(res)) print("Alternate optimization cutoff") start_time = time.time() query_engine = index.as_query_engine( node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.7)] ) res = query_engine.query("What is the population of Berlin?") end_time = time.time() print("Total time elapsed: {}".format(end_time - start_time)) print("Answer: {}".format(res))
    Without optimization
    INFO:root:> [query] Total LLM token usage: 3545 tokens
    INFO:root:> [query] Total embedding token usage: 7 tokens
    Total time elapsed: 2.8928110599517822
    Answer:
    The population of Berlin in 1949 was approximately 2.2 million inhabitants. After the fall of the Berlin Wall in 1989, the population of Berlin increased to approximately 3.7 million inhabitants.
    With optimization
    INFO:root:> [optimize] Total embedding token usage: 7 tokens
    INFO:root:> [query] Total LLM token usage: 1779 tokens
    INFO:root:> [query] Total embedding token usage: 7 tokens
    Total time elapsed: 2.346346139907837
    Answer:
    The population of Berlin is around 4.5 million.
    Alternate optimization cutoff
    INFO:root:> [optimize] Total embedding token usage: 7 tokens
    INFO:root:> [query] Total LLM token usage: 3215 tokens
    INFO:root:> [query] Total embedding token usage: 7 tokens
    Total time elapsed: 2.101111888885498
    Answer:
    The population of Berlin is around 4.5 million.
Back to top
Previous
NVIDIA NIMs
Next
PII Masking
```

## Response

```python
import os
os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
from llama_index.core import download_loader
from llama_index.readers.wikipedia import WikipediaReader
loader = WikipediaReader()
documents = loader.load_data(pages=["Berlin"])
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
index = VectorStoreIndex.from_documents(documents)
print("Without optimization")
start_time = time.time()
query_engine = index.as_query_engine()
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))
print("With optimization")
start_time = time.time()
query_engine = index.as_query_engine(
    node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)]
)
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))
print("Alternate optimization cutoff")
start_time = time.time()
query_engine = index.as_query_engine(
    node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.7)]
)
res = query_engine.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))
```
""".strip()
aim_value = Text(value)
run.track(aim_value, **track_settings)
