from typing import List, Dict, Tuple, TypedDict
import logging
from textblob.en.sentiments import PatternAnalyzer
from textblob.en.taggers import NLTKTagger
from textblob.en.np_extractors import FastNPExtractor
from textblob.tokenizers import SentenceTokenizer, WordTokenizer
import os
from typing import List

from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

from jet.code.markdown_types.markdown_parsed_types import HeaderDoc
from jet.code.markdown_utils import parse_markdown
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType
from jet.utils.print_utils import print_dict_types
from jet.vectors.semantic_search.header_vector_search import search_headers
from jet.wordnet.sentence import split_sentences


def get_noun_phrases(texts: List[str]) -> List[List[str]]:
    noun_phrases: List[List[str]] = []
    extractor = ConllExtractor()
    for text in texts:
        textblob = TextBlob(text, np_extractor=None)
        noun_phrases.append(list(textblob.noun_phrases))
    return noun_phrases


class Chunk(TypedDict):
    """Type definition for a processed text chunk."""
    text: str
    noun_phrases: List[str]
    pos_tags: List[Tuple[str, str]]
    sentiment: Dict[str, float]


class RAGPreprocessor:
    """Preprocesses text for RAG embeddings search using TextBlob."""

    def __init__(
        self,
        tokenizer=SentenceTokenizer(),
        np_extractor=FastNPExtractor(),
        pos_tagger=NLTKTagger(),
        analyzer=PatternAnalyzer()
    ):
        self.tokenizer = tokenizer
        self.np_extractor = np_extractor
        self.pos_tagger = pos_tagger
        self.analyzer = analyzer
        self.word_tokenizer = WordTokenizer()

    def create_blob(self, text: str, word_tokenize: bool = False) -> TextBlob:
        """Creates a TextBlob instance with configured settings."""
        tokenizer = self.word_tokenizer if word_tokenize else self.tokenizer
        return TextBlob(
            text,
            tokenizer=tokenizer,
            np_extractor=self.np_extractor,
            pos_tagger=self.pos_tagger,
            analyzer=self.analyzer
        )

    def preprocess_for_rag(self, document: str) -> List[Chunk]:
        """
        Preprocesses a document into chunks for RAG embeddings search.

        Args:
            document: Raw text document to process.

        Returns:
            List of dictionaries containing text chunks and metadata.
        """
        blob = self.create_blob(
            document)  # Use SentenceTokenizer for splitting
        chunks: List[Chunk] = []

        for sentence in blob.sentences:
            sentence_text = str(sentence)
            # Use WordTokenizer for sentence-level processing
            sentence_blob = self.create_blob(sentence_text, word_tokenize=True)

            chunk: Chunk = {
                "text": sentence_text,
                "noun_phrases": sentence_blob.noun_phrases,
                "pos_tags": sentence_blob.tags,
                "sentiment": {
                    "polarity": sentence_blob.sentiment.polarity,
                    "subjectivity": sentence_blob.sentiment.subjectivity
                }
            }
            chunks.append(chunk)

        return chunks


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/code/extraction/generated/run_extract_notebook_texts/GenAI_Agents/docs/Academic_Task_Learning_Agent_LangGraph.md"
    md_content: str = load_file(md_file)
    save_file(md_content, f"{output_dir}/md_content.md")

    query = "Long context agent summary"
    embed_model: EmbedModelType = "all-MiniLM-L6-v2"
    top_k = None
    threshold = 0.0
    chunk_size = 500
    chunk_overlap = 100
    merge_chunks = False

    markdown_tokens = base_parse_markdown(md_content)
    save_file(markdown_tokens, f"{output_dir}/markdown_tokens.json")

    docs: List[HeaderDoc] = derive_by_header_hierarchy(
        md_content, ignore_links=True)
    save_file(docs, f"{output_dir}/docs.json")

    texts = [f"{doc["header"]}\n{doc["content"]}" for doc in docs]
    noun_phrases: List[List[str]] = get_noun_phrases(texts)
    save_file(noun_phrases, f"{output_dir}/noun_phrases.json")

    sentences: List[str] = [
        sentence for text in texts for sentence in split_sentences(text)]
    save_file(sentences, f"{output_dir}/sentences.json")

    sentences: List[str] = [
        blob.sentences for text in texts for blob in TextBlob(text)]
    save_file(sentences, f"{output_dir}/sentences_2.json")

    search_results = list(
        search_headers(
            docs,
            query,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer_model=embed_model,
            merge_chunks=merge_chunks
        )
    )

    save_file(search_results, f"{output_dir}/search_results.json")
