from bs4 import BeautifulSoup
import spacy
from typing import List, Tuple
import numpy as np

from jet.transformers.formatters import format_json
from jet.logger import logger

# Load Medium SpaCy Model with word vectors
nlp = spacy.load("en_core_web_md")


def extract_text_from_html(html_content: str) -> str:
    """
    Extracts and cleans text from raw HTML.

    Args:
        html_content (str): Raw HTML content.

    Returns:
        str: Cleaned text extracted from HTML.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def compute_similarity(text1: str, text2: str) -> float:
    """
    Computes semantic similarity between two texts using SpaCy word vectors.

    Args:
        text1 (str): First text.
        text2 (str): Second text.

    Returns:
        float: Similarity score (0 to 1).
    """
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)


def merge_similar_sao(sao_list: List[Tuple[str, List[str], List[str]]], threshold: float = 0.85) -> List[Tuple[str, List[str], List[str]]]:
    """
    Merges similar SAO (Subject-Action-Object) sets based on semantic similarity.

    Args:
        sao_list (List[Tuple[str, List[str], List[str]]]): List of extracted SAO tuples.
        threshold (float): Similarity threshold for merging.

    Returns:
        List[Tuple[str, List[str], List[str]]]: Merged SAO list.
    """
    merged_sao = []

    for subject, actions, objects in sao_list:
        found = False

        for i, (m_subject, m_actions, m_objects) in enumerate(merged_sao):
            if compute_similarity(subject, m_subject) > threshold:
                merged_sao[i] = (
                    m_subject,
                    sorted(set(m_actions + actions)),
                    sorted(set(m_objects + objects))
                )
                found = True
                break

        if not found:
            merged_sao.append(
                (subject, sorted(set(actions)), sorted(set(objects))))

    return merged_sao


def extract_subject_action_objects(text: str) -> List[Tuple[str, List[str], List[str]]]:
    """
    Extracts multiple subjects, actions (verbs), and objects from a given text.
    Normalizes verbs (lemmatization), sorts actions & objects, and merges similar SAOs.

    Args:
        text (str): Input text.

    Returns:
        List[Tuple[str, List[str], List[str]]]: List of (subject, [actions], [objects])
    """
    doc = nlp(text)

    sao_tuples = []
    subjects = []
    actions = []
    objects = []

    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            if subjects:
                # Store previous set if a new subject is found
                sao_tuples.append(
                    (subjects[-1], sorted(set(actions)), sorted(set(objects))))
                actions.clear()
                objects.clear()
            subjects.append(token.lemma_)  # Use lemma to normalize subject
        elif token.pos_ == "VERB":
            actions.append(token.lemma_)  # Use lemma for verb normalization
        elif token.dep_ in {"dobj", "attr", "pobj"}:
            objects.append(token.lemma_)  # Use lemma for object normalization

    # Capture the last set
    if subjects:
        sao_tuples.append(
            (subjects[-1], sorted(set(actions)), sorted(set(objects))))

    return merge_similar_sao(sao_tuples)


def extract_sao_from_html(html_content: str) -> List[Tuple[str, List[str], List[str]]]:
    """
    Extracts multiple Subject-Action-Object (SAO) sets from unstructured HTML.

    Args:
        html_content (str): Raw HTML content.

    Returns:
        List[Tuple[str, List[str], List[str]]]: List of (subject, [actions], [objects])
    """
    text = extract_text_from_html(html_content)
    return extract_subject_action_objects(text)


if __name__ == "__main__":
    # sample_html = "<html><body><p>The CEO announced the new policy and implemented changes.</p></body></html>"
    sample_html = "<html><body><p>The one who announced the new policy and implemented changes is the CEO.</p></body></html>"
    result = extract_sao_from_html(sample_html)

    logger.newline()
    logger.success(format_json(result))
