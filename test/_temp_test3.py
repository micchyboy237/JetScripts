from unstructured.ingest.v2.pipeline.pipeline import Pipeline

# Single-call RAG-ready output: partitions → chunks → embeds → cleans
pipeline = Pipeline.from_config(
    {
        "processor": {
            "type": "partition",
            "strategy": "hi_res",
            "chunking_strategy": "by_title",
            "chunk_max_characters": 1024,
            "embedding_model": "text-embedding-3-small",
            "prompt_configs": [
                {
                    "template": "Extract 3 key topics: {text}",
                    "field": "topics",
                    "element_types": ["NarrativeText"],
                }
            ],
        },
        "source": {"type": "local", "path": "./docs"},
        "destination": {"type": "local", "output_dir": "./rag_output"},
    }
)

pipeline.run()
# Output: JSON files with text, embeddings, summaries, and metadata ready for vector DB upsert
