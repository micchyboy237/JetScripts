# Search documents by query
curl -X POST "http://0.0.0.0:8000/search" \
  -H "Content-Type: application/json" \
  -H "Accept: application/x-ndjson" \
  -d '{
    "query": "List all ongoing and upcoming isekai anime 2025",
    "top_k": 10,
    "embed_model": "static-retrieval-mrl-en-v1",
    "llm_model": "llama-3.2-1b-instruct-4bit",
    "seed": 42,
    "use_cache": true,
    "min_mtld": 100.0,
    "stream": true
  }'
