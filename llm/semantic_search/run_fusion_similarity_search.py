from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.file.utils import load_file
from jet.llm.ollama.base import Ollama, OllamaEmbedding
from jet.logger import logger
from jet.scrapers.preprocessor import html_to_markdown
from jet.token.token_utils import get_model_max_tokens
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.similarity import fuse_similarity_scores, get_query_similarity_scores
from llama_index.core.schema import NodeWithScore, TextNode


LLM_MODEL = "gemma3:4b"
LLM_MAX_TOKENS = get_model_max_tokens(LLM_MODEL)
EMBED_MODEL = "mxbai-embed-large"
EMBED_MODEL_2 = "paraphrase-multilingual"
EMBED_MODEL_3 = "granite-embedding"
EVAL_MODEL = "gemma3:4b"

DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/valid-ids-scraper/philippines_national_id_registration_tips_2025/scraped_html.html"
OUTPUT_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_llm_reranker"

query = "What are the steps in registering a National ID in the Philippines?"
html: str = load_file(DATA_FILE)

llm = Ollama(temperature=0.3, model=LLM_MODEL,
             request_timeout=300.0, context_window=LLM_MAX_TOKENS)
embed_model = OllamaEmbedding(model_name=EMBED_MODEL)


md_text = html_to_markdown(html)
header_contents = get_md_header_contents(md_text)

all_nodes = [TextNode(text=h["content"], metadata={
                      "doc_index": idx}) for idx, h in enumerate(header_contents)]
all_texts = [node.text for node in all_nodes]
all_texts_dict = {node.text: node for node in all_nodes}


query_similarities = get_query_similarity_scores(
    query, all_texts, model_name=EMBED_MODEL)
query_similarities_2 = get_query_similarity_scores(
    query, all_texts, model_name=EMBED_MODEL_2)
query_similarities_3 = get_query_similarity_scores(
    query, all_texts, model_name=EMBED_MODEL_3)
vector_results = fuse_similarity_scores(
    query_similarities, query_similarities_2, query_similarities_3)
# nodes_with_scores = [
#     NodeWithScore(
#         node=TextNode(text=text,
#                       metadata=all_texts_dict[text].metadata),
#         score=score
#     )
#     for text, score in query_similarities[0]["results"].items()
# ]
copy_to_clipboard(vector_results)
logger.success(format_json(vector_results))
