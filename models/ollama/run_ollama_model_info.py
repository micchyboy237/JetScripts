# Run the script and print the result
from jet.llm.models import build_ollama_model_contexts, build_ollama_model_embeddings, build_ollama_hf_mappings
from jet.logger import logger
from jet.transformers.formatters import format_json


if __name__ == "__main__":
    ollama_model_contexts = build_ollama_model_contexts()
    logger.newline()
    logger.debug(f"ollama_model_contexts ({len(ollama_model_contexts)})")
    logger.success(format_json(ollama_model_contexts))

    ollama_model_embeddings = build_ollama_model_embeddings()
    logger.newline()
    logger.debug(f"ollama_model_embeddings ({len(ollama_model_embeddings)})")
    logger.success(format_json(ollama_model_embeddings))

    ollama_model_names = list(ollama_model_contexts.keys())
    logger.newline()
    logger.debug(f"ollama_model_names ({len(ollama_model_names)})")
    logger.success(format_json(ollama_model_names))

    ollama_model_hf_mappings = build_ollama_hf_mappings()
    logger.newline()
    logger.debug(f"ollama_model_hf_mappings ({len(ollama_model_hf_mappings)})")
    logger.success(format_json(ollama_model_hf_mappings))
