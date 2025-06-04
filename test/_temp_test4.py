import re
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from scipy.stats import entropy

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        """
        Extract keywords from the query by removing stop words.
        """
        stop_words = {"a", "an", "the", "is", "are",
                      "in", "on", "of", "to", "what", "how"}
        words = re.findall(r'\w+', query.lower())
        return [word for word in words if word not in stop_words]

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML tags, extra whitespace, and special characters.
        """
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
        """
        Compute relevance score using semantic similarity and keyword matching.

        Args:
            text: The document text.
            header: The section header (optional, for keyword matching).
        Returns:
            Tuple of (cosine similarity score, keyword presence boolean)
        """
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
        """
        Classify content as promotional using DistilBERT.

        Args:
            text: The document text.
            header: The section header.
        Returns:
            Tuple of (is_promotional, confidence score)
        """
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
        """
        Compute text entropy to detect boilerplate or repetitive content.

        Args:
            text: The cleaned document text.
        Returns:
            Entropy value (higher indicates more varied content).
        """
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
        """
        Tag a single document and return its relevance score.

        Args:
            doc: Dict with parent_header, header, and text.
        Returns:
            Dict with parent_header, header, tag, and score.
        """
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
        """
        Tag documents and return relevance scores with hierarchical structure.

        Args:
            documents: List of dicts with parent_header, header, and text.
        Returns:
            List of dicts with parent_header, header, tag, and score.
        """
        for doc in documents:
            if not isinstance(doc, dict) or "header" not in doc or "text" not in doc:
                raise ValueError(
                    "Each document must be a dict with 'header' and 'text' keys")

        contents = [f"{self.clean_text(doc.get('header', ''))} {self.clean_text(doc.get('text', ''))}".strip(
        ) for doc in documents]
        valid_contents = [c for c in contents if c]
        content_embeddings = None
        if valid_contents:
            try:
                content_embeddings = self.model.encode(
                    valid_contents, convert_to_tensor=True, device=self.device, batch_size=32)
            except Exception as e:
                print(f"Error batch encoding: {str(e)}")

        results = []
        embedding_idx = 0
        for doc, content in zip(documents, contents):
            parent_header = doc.get("parent_header", None)
            header = doc.get("header", "")
            text = doc.get("text", "")
            cleaned_text = self.clean_text(text)
            cleaned_header = self.clean_text(header)

            if not content:
                results.append({
                    "parent_header": parent_header,
                    "header": header,
                    "tag": "Irrelevant",
                    "score": 0.0
                })
                continue

            try:
                if content_embeddings is not None and content in valid_contents:
                    content_embedding = content_embeddings[embedding_idx]
                    embedding_idx += 1
                    cosine_score = util.cos_sim(
                        self.query_embedding, content_embedding).item()
                else:
                    cosine_score = 0.0

                has_keywords = any(
                    keyword in f"{cleaned_text} {cleaned_header}" for keyword in self.keywords)
                tag = "Relevant" if cosine_score >= self.threshold or has_keywords else "Irrelevant"
                final_score = cosine_score if cosine_score >= self.threshold else 0.0 if has_keywords else cosine_score

                results.append({
                    "parent_header": parent_header,
                    "header": header,
                    "tag": tag,
                    "score": final_score
                })
            except Exception as e:
                print(f"Error tagging document {header}: {str(e)}")
                results.append({
                    "parent_header": parent_header,
                    "header": header,
                    "tag": "Irrelevant",
                    "score": 0.0
                })

        return results

    def analyze_noise(self, documents: List[Dict[str, Optional[str]]]) -> Dict[str, List[Dict]]:
        """
        Analyze which documents are noise based on tagging, classifier, and heuristics.

        Args:
            documents: List of dicts with parent_header, header, and text.
        Returns:
            Dict with 'noise' and 'relevant' lists of document results, including noise reasons.
        """
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
                # Semantic similarity
                if doc["score"] < self.threshold:
                    reasons.append(
                        f"Low semantic similarity (score: {doc['score']:.3f})")
                # Keyword absence
                has_keywords = any(
                    keyword in content for keyword in self.keywords)
                if not has_keywords:
                    reasons.append("No query-related keywords")
                # Promotional content
                is_promotional, promo_score = self._is_promotional(
                    orig_doc.get("text", ""), doc["header"])
                if is_promotional:
                    reasons.append(
                        f"Promotional content (confidence: {promo_score:.3f})")
                # Content length
                word_count = len(cleaned_text.split())
                if word_count < 10:
                    reasons.append(
                        f"Short content length ({word_count} words)")
                # Boilerplate detection via entropy
                text_entropy = self._compute_entropy(cleaned_text)
                if text_entropy < 1.0 and word_count > 0:
                    reasons.append(
                        f"Low text entropy ({text_entropy:.3f}, likely boilerplate)")
                # Header analysis
                if any(term in cleaned_header for term in ["nav", "menu", "footer", "sidebar"]):
                    reasons.append("Header indicates structural content")
                result["noise_reasons"] = reasons
                noise.append(result)
            else:
                relevant.append(result)

        return {"noise": noise, "relevant": relevant}


if __name__ == "__main__":
    from jet.file.utils import load_file, save_file
    from jet.token.token_utils import split_headers
    from jet.vectors.document_types import HeaderDocument

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    docs = [HeaderDocument(**doc) for doc in docs]
    chunked_docs = split_headers(docs)
    documents = [
        {
            "parent_header": doc["metadata"]["parent_header"],
            "header": doc["metadata"]["header"],
            "text": doc["content"]
        }
        for doc in chunked_docs
    ]
    query = "List all ongoing and upcoming isekai anime 2025."

    # Initialize tagger
    tagger = DocumentTagger(query, threshold=0.5)

    # Print device being used
    print(f"Using device: {tagger.device}")

    # Tag documents
    tagged = tagger.tag_documents(documents)

    # Sort by score (highest to lowest), even if score is 0
    tagged.sort(key=lambda x: x["score"], reverse=True)

    # Filter: keep only items tagged as relevant
    filtered_tagged = [tag for tag in tagged if tag["score"] > 0.5]

    # Display results
    for result in filtered_tagged:
        parent = result["parent_header"] or "None"
        print(
            f"Parent: {parent}, Header: {result['header']}, Tag: {result['tag']} (Score: {result['score']:.3f})")

    save_file(tagged, f"{output_dir}/tagged.json")
    save_file(filtered_tagged, f"{output_dir}/filtered_tagged.json")
    save_file([tag for tag in tagged if tag["tag"] == "relevant"],
              f"{output_dir}/relevant.json")
    save_file([tag for tag in tagged if tag["tag"] == "irrelevant"],
              f"{output_dir}/irrelevant.json")
