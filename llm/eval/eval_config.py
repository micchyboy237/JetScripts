from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.ollama.base import Ollama

# LLM and embedding config

base_url = "http://localhost:11434",
llm_settings = {
    "model": "llama3.1",
    "context_window": 4096,
    "request_timeout": 300.0,
    "temperature": 0,
}
embed_settings = {
    "model": "nomic-embed-text",
    "chunk_size": 768,
    "chunk_overlap": 75,
}

llm = Ollama(
    temperature=llm_settings['temperature'],
    context_window=llm_settings['context_window'],
    request_timeout=llm_settings['request_timeout'],
    model=llm_settings['model'],
    base_url=base_url,
)
embedding = OllamaEmbedding(
    model_name=embed_settings['model'],
    base_url=base_url,
)
