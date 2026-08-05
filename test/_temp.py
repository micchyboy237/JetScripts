from unstructured.chunking.title import chunk_by_title

# Alternative: from unstructured.chunking.semantic import chunk_semantic

chunks = chunk_by_title(
    elements=elements,
    max_characters=1024,
    overlap=128,  # Token overlap for context continuity
    multipage_sections=True,  # Don't break sections across pages
    combine_text_under_n_chars=200,  # Merge small consecutive elements
)

print(f"Partitioned {len(elements)} elements → {len(chunks)} chunks")
print(f"First chunk metadata keys: {list(chunks[0].metadata.to_dict().keys())}")
