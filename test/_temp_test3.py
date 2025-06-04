import re
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from scipy.stats import entropy
import spacy
from spacy.language import Language
from span_marker import SpanMarkerModel

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # Register SpanMarker as a custom SpaCy component
# @Language.factory(
#     "span_marker",
#     default_config={"model_name": "tomaarsen/span-marker-mbert-base-multinerd"}
# )
# def create_span_marker_component(nlp: Language, name: str, model_name: str):
#     """
#     Factory function to create a SpanMarker component for SpaCy pipeline.

#     Args:
#         nlp: SpaCy Language object.
#         name: Name of the pipeline component.
#         model_name: Pretrained SpanMarker model name.
#     Returns:
#         SpanMarker component for SpaCy pipeline.
#     """
#     from span_marker import SpanMarkerModel as SpanMarker
#     model = SpanMarker.from_pretrained(model_name)
#     return SpanMarker(model)


class DocumentTagger:
    def __init__(self, query: str, keywords: List[str] = None, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.3):
        """
        Initialize the tagger with a query, optional keywords, model, and relevance threshold.

        Args:
            query: The search query.
            keywords: Optional list of keywords; if None, derived from query.
            model_name: SentenceTransformer model for semantic similarity.
            threshold: Cosine similarity threshold for relevance (0 to 1).
        """
        self.query = query.lower()
        self.keywords = keywords or self._extract_keywords(query)
        self.threshold = threshold
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.classifier = pipeline(
                "text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=self.device)
        except Exception as e:
            raise ValueError(f"Error loading models: {str(e)}")
        self.query_embedding = self.model.encode(
            self.query, convert_to_tensor=True, device=self.device)

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {"a", "an", "the", "is", "are",
                      "in", "on", "of", "to", "what", "how"}
        words = re.findall(r'\w+', query.lower())
        return [word for word in words if word not in stop_words]

    def clean_text(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                return ""
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = ' '.join(text.split())
            return text.lower()
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return ""

    def _compute_relevance(self, text: str, header: str = "") -> Tuple[float, bool]:
        cleaned_text = self.clean_text(text)
        cleaned_header = self.clean_text(header)
        if not cleaned_text and not cleaned_header:
            return 0.0, False

        try:
            content = f"{cleaned_header} {cleaned_text}".strip()
            if not content:
                return 0.0, False
            content_embedding = self.model.encode(
                content, convert_to_tensor=True, device=self.device)
            cosine_score = util.cos_sim(
                self.query_embedding, content_embedding).item()
        except Exception as e:
            print(f"Error computing embedding: {str(e)}")
            cosine_score = 0.0

        has_keywords = any(
            keyword in f"{cleaned_text} {cleaned_header}" for keyword in self.keywords)
        return cosine_score, has_keywords

    def _is_promotional(self, text: str, header: str = "") -> Tuple[bool, float]:
        try:
            content = f"{self.clean_text(header)} {self.clean_text(text)}".strip(
            )
            if not content:
                return False, 0.0
            result = self.classifier(content, truncation=True, max_length=512)
            is_promotional = result[0]["label"] == "POSITIVE" and result[0]["score"] > 0.7
            return is_promotional, result[0]["score"]
        except Exception as e:
            print(f"Error classifying promotional content: {str(e)}")
            return False, 0.0

    def _compute_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        probs = np.array(list(word_counts.values())) / len(words)
        return entropy(probs) if probs.size > 0 else 0.0

    def _tag_document(self, doc: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        try:
            parent_header = doc.get("parent_header", None)
            header = doc.get("header", "")
            text = doc.get("text", "")
            if not header or not text:
                return {
                    "parent_header": parent_header,
                    "header": header,
                    "tag": "Irrelevant",
                    "score": 0.0
                }

            score, has_keywords = self._compute_relevance(text, header)
            tag = "Relevant" if score >= self.threshold or has_keywords else "Irrelevant"
            final_score = score if score >= self.threshold else 0.0 if has_keywords else score

            return {
                "parent_header": parent_header,
                "header": header,
                "tag": tag,
                "score": final_score
            }
        except Exception as e:
            print(
                f"Error tagging document {doc.get('header', 'unknown')}: {str(e)}")
            return {
                "parent_header": doc.get("parent_header", None),
                "header": doc.get("header", ""),
                "tag": "Irrelevant",
                "score": 0.0
            }

    def tag_documents(self, documents: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
        for doc in documents:
            if not isinstance(doc, dict) or "header" not in doc or "text" not in doc:
                raise ValueError(
                    "Each document must be a dict with 'header' and 'text' keys")

        results = []
        for doc in documents:
            results.append(self._tag_document(doc))
        return results

    def analyze_noise(self, documents: List[Dict[str, Optional[str]]]) -> Dict[str, List[Dict]]:
        tagged_docs = self.tag_documents(documents)
        noise = []
        relevant = []

        for doc, orig_doc in zip(tagged_docs, documents):
            cleaned_text = self.clean_text(orig_doc.get("text", ""))
            cleaned_header = self.clean_text(doc["header"])
            content = f"{cleaned_header} {cleaned_text}".strip()

            result = {
                "parent_header": doc["parent_header"],
                "header": doc["header"],
                "text": orig_doc.get("text", ""),
                "tag": doc["tag"],
                "score": doc["score"]
            }

            if doc["tag"] == "Irrelevant":
                reasons = []
                if doc["score"] < self.threshold:
                    reasons.append(
                        f"Low semantic similarity (score: {doc['score']:.3f})")
                has_keywords = any(
                    keyword in content for keyword in self.keywords)
                if not has_keywords:
                    reasons.append("No query-related keywords")
                is_promotional, promo_score = self._is_promotional(
                    orig_doc.get("text", ""), doc["header"])
                if is_promotional:
                    reasons.append(
                        f"Promotional content (confidence: {promo_score:.3f})")
                word_count = len(cleaned_text.split())
                if word_count < 10:
                    reasons.append(
                        f"Short content length ({word_count} words)")
                text_entropy = self._compute_entropy(cleaned_text)
                if text_entropy < 1.0 and word_count > 0:
                    reasons.append(
                        f"Low text entropy ({text_entropy:.3f}, likely boilerplate)")
                if any(term in cleaned_header for term in ["nav", "menu", "footer", "sidebar"]):
                    reasons.append("Header indicates structural content")
                result["noise_reasons"] = reasons
                noise.append(result)
            else:
                relevant.append(result)

        return {"noise": noise, "relevant": relevant}


def process_batch_with_nlp(nlp, documents: List[Dict[str, Optional[str]]], batch_size: int = 10) -> List[Dict[str, Optional[str]]]:
    """
    Process documents in batches using the SpaCy NLP pipeline, skipping invalid inputs.

    Args:
        nlp: SpaCy pipeline with SpanMarker.
        documents: List of documents with header and text.
        batch_size: Number of documents to process per batch.
    Returns:
        List of processed documents with entity annotations.
    """
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        valid_texts = []
        valid_docs = []
        for doc in batch:
            text = f"{doc.get('header', '')} {doc.get('text', '')}".strip()
            if text and len(text) > 0:
                valid_texts.append(text)
                valid_docs.append(doc)
            else:
                print(
                    f"Skipping empty or invalid document: {doc.get('header', 'unknown')}")
                results.append({
                    "parent_header": doc.get("parent_header", None),
                    "header": doc.get("header", ""),
                    "text": doc.get("text", ""),
                    "entities": [],
                    "tag": "Irrelevant",
                    "score": 0.0
                })

        if not valid_texts:
            continue

        try:
            print(
                f"Processing batch {i // batch_size + 1} with {len(valid_texts)} documents")
            batch_docs = list(nlp.pipe(valid_texts))
            for doc, orig_doc in zip(batch_docs, valid_docs):
                entities = [{"start": ent.start_char, "end": ent.end_char,
                             "label": ent.label_} for ent in doc.ents]
                results.append({
                    "parent_header": orig_doc.get("parent_header", None),
                    "header": orig_doc.get("header", ""),
                    "text": orig_doc.get("text", ""),
                    "entities": entities,
                    "tag": "Processed",
                    "score": 0.0
                })
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {str(e)}")
            for doc in valid_docs:
                print(
                    f"Skipping document due to error: {doc.get('header', 'unknown')}")
                results.append({
                    "parent_header": doc.get("parent_header", None),
                    "header": doc.get("header", ""),
                    "text": doc.get("text", ""),
                    "entities": [],
                    "tag": "Irrelevant",
                    "score": 0.0
                })

    return results


if __name__ == "__main__":
    from jet.file.utils import load_file, save_file
    from jet.token.token_utils import split_headers
    from jet.vectors.document_types import HeaderDocument

    # Load SpaCy with SpanMarker
    print("Loading spaCy model 'en_core_web_sm'")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        print("Adding sentencizer pipeline")
        nlp.add_pipe("sentencizer")
        print("Adding SpanMarker pipeline")
        nlp.add_pipe("span_marker")
    except Exception as e:
        print(f"Error setting up SpaCy pipeline: {str(e)}")
        raise

    # Load documents
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    print(f"Loading JSON data from: {docs_file}")
    docs = load_file(docs_file)
    print(f"Loaded {len(docs)} JSON documents")
    docs = [HeaderDocument(**doc) for doc in docs]
    print("Converting to HeaderDocument and splitting headers")
    chunked_docs = split_headers(docs)
    print(f"Created {len(chunked_docs)} chunked documents")

    documents = [
        {
            "parent_header": doc["metadata"]["parent_header"],
            "header": doc["metadata"]["header"],
            "text": doc["content"]
        }
        for doc in chunked_docs
    ]
    print(f"Preparing {len(documents)} documents")

    # Process documents with NLP pipeline
    processed_docs = process_batch_with_nlp(nlp, documents, batch_size=10)

    # Initialize tagger
    query = "List all ongoing and upcoming isekai anime 2025."
    tagger = DocumentTagger(query, threshold=0.5)

    # Print device being used
    print(f"Using device: {tagger.device}")

    # Tag documents
    tagged = tagger.tag_documents(processed_docs)

    # Sort by score (highest to lowest)
    tagged.sort(key=lambda x: x["score"], reverse=True)

    # Filter: keep only items tagged as relevant
    filtered_tagged = [tag for tag in tagged if tag["score"] > 0.5]

    # Display results
    for result in filtered_tagged:
        parent = result["parent_header"] or "None"
        print(
            f"Parent: {parent}, Header: {result['header']}, Tag: {result['tag']} (Score: {result['score']:.3f})")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_file(tagged, f"{output_dir}/tagged.json")
    save_file(filtered_tagged, f"{output_dir}/filtered_tagged.json")
    save_file([tag for tag in tagged if tag["tag"] == "Relevant"],
              f"{output_dir}/relevant.json")
    save_file([tag for tag in tagged if tag["tag"] == "Irrelevant"],
              f"{output_dir}/irrelevant.json")
