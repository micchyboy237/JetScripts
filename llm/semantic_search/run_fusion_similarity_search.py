import os
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.file.utils import load_file, save_file
from jet.llm.mlx.utils.base import get_model_max_tokens
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.scrapers.preprocessor import html_to_markdown
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.similarity import query_similarity_scores
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore, TextNode


LLM_MODEL = "gemma3:4b"
LLM_MAX_TOKENS = (LLM_MODEL)
EMBED_MODEL = "mxbai-embed-large"
EMBED_MODEL_2 = "paraphrase-multilingual"
EMBED_MODEL_3 = "granite-embedding"
EMBED_MODELS = [
    EMBED_MODEL,
    EMBED_MODEL_2,
    EMBED_MODEL_3,
]
EVAL_MODEL = "gemma3:4b"

DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


docs: list[dict] = load_file(DATA_FILE)
query = "List popular isekai reincarnation anime series released in the last few years, focusing on those with high ratings and recent popularity."
top_k = 5

# all_nodes = [TextNode(text=h["content"], metadata={
#                       "doc_index": idx}) for idx, h in enumerate(header_contents)]
header_docs = [Document(text=doc["text"], metadata={
    "doc_index": doc["metadata"]["doc_index"]}) for doc in docs]

chunk_overlap = 40
chunk_size = 300
splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
all_nodes: list[TextNode] = splitter.get_nodes_from_documents(
    documents=header_docs)

# Build lookup of doc_index -> original text
doc_index_to_text = {
    doc.metadata["doc_index"]: docs[doc.metadata["doc_index"]]["text"] for doc in all_nodes}


all_texts = [node.text for node in all_nodes]
all_texts_dict = {node.text: node for node in all_nodes}

query_similarities = query_similarity_scores(
    query, all_texts, model=[EMBED_MODEL, EMBED_MODEL_2, EMBED_MODEL_3])

nodes_with_scores = []
seen_docs = set()  # To track unique texts
for result in query_similarities:
    text = result["text"]
    score = result["score"]
    doc_index = all_texts_dict[text].metadata["doc_index"]
    if doc_index not in seen_docs:  # Ensure unique by text
        seen_docs.add(doc_index)
        nodes_with_scores.append(
            NodeWithScore(
                node=TextNode(text=header_docs[doc_index].text,
                              metadata=all_texts_dict[text].metadata),
                score=score
            )
        )
copy_to_clipboard(nodes_with_scores)
display_jet_source_nodes(query, nodes_with_scores[:top_k])


output_path = f"{OUTPUT_DIR}/nodes_with_scores.json"
save_file(nodes_with_scores, output_path)
print(f"Save JSON data to: {output_path}")
