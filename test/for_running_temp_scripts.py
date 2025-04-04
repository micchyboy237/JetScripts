from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.file.utils import load_file
from jet.llm.ollama.base import Ollama, OllamaEmbedding
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.scrapers.preprocessor import html_to_markdown
from jet.token.token_utils import get_model_max_tokens
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.similarity import fuse_similarity_scores, get_query_similarity_scores
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore, TextNode


LLM_MODEL = "gemma3:4b"
LLM_MAX_TOKENS = get_model_max_tokens(LLM_MODEL)
EMBED_MODEL = "mxbai-embed-large"
EMBED_MODEL_2 = "paraphrase-multilingual"
EMBED_MODEL_3 = "granite-embedding"
EMBED_MODELS = [
    EMBED_MODEL,
    EMBED_MODEL_2,
    EMBED_MODEL_3,
]
EVAL_MODEL = "gemma3:4b"

DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/valid-ids-scraper/philippines_national_id_registration_tips_2025/scraped_html.html"
OUTPUT_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_llm_reranker"

html: str = load_file(DATA_FILE)
query = "What are the steps in registering a National ID in the Philippines?"
top_k = 5

llm = Ollama(temperature=0.3, model=LLM_MODEL,
             request_timeout=300.0, context_window=LLM_MAX_TOKENS)
embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

md_text = html_to_markdown(html)
header_contents = get_md_header_contents(md_text)

# all_nodes = [TextNode(text=h["content"], metadata={
#                       "doc_index": idx}) for idx, h in enumerate(header_contents)]
header_docs = [Document(text=h["content"], metadata={
    "doc_index": idx}) for idx, h in enumerate(header_contents)]
chunk_overlap = 40
chunk_size = min(get_model_max_tokens(embed_model)
                 for embed_model in EMBED_MODELS)
splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
all_nodes: list[TextNode] = splitter.get_nodes_from_documents(
    documents=header_docs)

# Build lookup of doc_index -> original text
doc_index_to_text = {doc.metadata["doc_index"]
    : doc.text for doc in header_docs}

# Inject start_idx and end_idx into each node's metadata
for node in all_nodes:
    doc_index = node.metadata["doc_index"]
    full_text = doc_index_to_text[doc_index]
    try:
        start_idx = full_text.index(node.text)
        end_idx = start_idx + len(node.text)
        node.metadata["start_idx"] = start_idx
        node.metadata["end_idx"] = end_idx
    except ValueError:
        logger.warning(
            f"Text not found in original doc for doc_index={doc_index}")
        node.metadata["start_idx"] = -1
        node.metadata["end_idx"] = -1
all_texts = [node.text for node in all_nodes]
all_texts_dict = {node.text: node for node in all_nodes}

query_similarities = get_query_similarity_scores(
    query, all_texts, model_name=[EMBED_MODEL, EMBED_MODEL_2, EMBED_MODEL_3])
nodes_with_scores = [
    NodeWithScore(
        node=TextNode(text=text,
                      metadata=all_texts_dict[text].metadata),
        score=score
    )
    for text, score in query_similarities[0]["results"].items()
]
copy_to_clipboard(nodes_with_scores)
display_jet_source_nodes(query, nodes_with_scores[:top_k])

# Run LLM Chat

top_node_texts = [
    f"Document number: {node.metadata['doc_index'] + 1}\nIndexes: {node.metadata['start_idx']} - {node.metadata['end_idx']}\n{node.text}"
    for node in nodes_with_scores[:top_k]
]
final_context = "\n\n".join(top_node_texts)
response = llm.chat(query, context=final_context)

copy_to_clipboard(response)
