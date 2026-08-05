from unstructured.transform.prompts import apply_prompt

# Generate concise summaries for each narrative chunk
summarized = apply_prompt(
    elements=chunks,
    prompt_template="Summarize this document section in ≤50 words:\n{text}",
    model_name="gpt-4o-mini",
    field_name="summary",  # Store result in element.metadata.summary
    include_element_types=["NarrativeText", "Title"],
)

print(f"Sample summary: {summarized[0].metadata.summary}")
