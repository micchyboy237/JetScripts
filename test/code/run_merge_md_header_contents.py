header_contents = merge_md_header_contents(
    [{"content": text} for text in node_texts],
    max_tokens=1500,
    tokenizer=get_ollama_tokenizer(LLM_MODEL).encode,
)
