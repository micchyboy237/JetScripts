import os
import spacy
import language_tool_python
import re
from jet.file.utils import load_file, save_file

nlp = spacy.load("en_core_web_sm")
grammar_tool = language_tool_python.LanguageTool("en-US")

data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/generated/search/top_anime_romantic_comedy_reddit_2024-2025/top_context_nodes.json"
data = load_file(data_file)


def clean_text(text):
    text = re.sub(r"#+|\*+|-+|=+|>", "", text)
    text = re.sub(r"[ \t]+(?=\n)", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


def is_grammatically_correct(sentence):
    matches = grammar_tool.check(sentence)
    major_errors = [m for m in matches if m.ruleIssueType in [
        "grammar", "misspelling"]]
    return len(major_errors) == 0


def extract_full_sentences(results):
    filtered_results = []

    for result in results:
        result_sentences = []
        # Split text by newlines to treat each line as a sentence
        lines = result["text"].split("\n")

        for line in lines:
            sentence = line.strip()
            if sentence:  # Ignore empty lines
                # Clean the sentence
                cleaned_sentence = clean_text(sentence)
                # Check if sentence is valid (length, punctuation, verb)
                if len(cleaned_sentence) > 10 and cleaned_sentence.endswith((".", "!", "?")) and any(token.pos_ == "VERB" for token in nlp(cleaned_sentence)):
                    if is_grammatically_correct(cleaned_sentence):
                        result_sentences.append(cleaned_sentence)

        # Only include results with at least one valid sentence
        if result_sentences:
            filtered_results.append({"text": "\n".join(result_sentences)})

    # Flatten sentences for output
    full_sentences = []
    for result in filtered_results:
        full_sentences.extend(result["text"].split("\n"))

    return full_sentences, filtered_results


filtered_sentences, filtered_results = extract_full_sentences(data["results"])

for sentence in filtered_sentences:
    print(sentence + "\n")

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
save_file({
    "all_count": len(data["results"]),
    "full_sentences_count": len(filtered_sentences),
    "full_sentences": filtered_sentences,
    "filtered_results_count": len(filtered_results),
    "filtered_results": filtered_results,
}, os.path.join(
    output_dir, "filtered_results.json"))
