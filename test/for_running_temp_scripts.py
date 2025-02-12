from typing import List, Dict, Any, TypedDict

from jet.token.token_utils import token_counter


class SummaryTokens(TypedDict):
    summary: str
    tokens: int


def group_summaries(
    summaries: list[str],
    max_tokens_per_group: int,
    tokenizer_model: str,
    separator: str = "\n\n\n"
) -> list[SummaryTokens]:
    grouped_summaries = []
    current_group = []
    current_tokens = 0

    for idx, summary in enumerate(summaries):
        prefixed_summary = f"Summary {idx + 1}\n\n{summary}"
        prefixed_summary_tokens: int = token_counter(
            prefixed_summary, tokenizer_model)

        combined_summary = separator.join(
            s["summary"] for s in current_group)
        combined_tokens: int = token_counter(
            combined_summary, tokenizer_model)
        next_group_tokens = prefixed_summary_tokens + combined_tokens

        if current_group and next_group_tokens >= max_tokens_per_group:
            grouped_summaries.append({
                "summary": combined_summary,
                "tokens": combined_tokens
            })
            current_group = []
            current_tokens = 0

        current_group.append(
            {"summary": prefixed_summary, "tokens": prefixed_summary_tokens})
        current_tokens += prefixed_summary_tokens

    if current_group:
        combined_summary = separator.join(s["summary"] for s in current_group)
        combined_tokens: int = token_counter(combined_summary, tokenizer_model)
        grouped_summaries.append({
            "summary": combined_summary,
            "tokens": combined_tokens
        })

    return grouped_summaries


# Example usage
if __name__ == "__main__":
    summaries = [
        "Test 1...",
        "Test 2...",
        "Test 3...",
        "Test 4...",
    ]
    max_tokens_per_group = 30
    tokenizer_model = "mistral"
    separator = "\n\n\n"

    grouped_summaries = group_summaries(
        summaries, max_tokens_per_group, tokenizer_model, separator)

    assert grouped_summaries == [
        {
            "summary": "Summary 1\n\nTest 1...\n\n\nSummary 2\n\nTest 2...",
            "tokens": 22
        },
        {
            "summary": "Summary 3\n\nTest 3...\n\n\nSummary 4\n\nTest 4...",
            "tokens": 22
        }
    ]
    print("Test passed.")
