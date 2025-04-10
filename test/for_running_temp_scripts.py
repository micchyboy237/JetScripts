import json
import os

from jet.features.scrape_search_chat import get_docs_from_html, get_nodes_from_docs, get_nodes_parent_mapping, rerank_nodes
from jet.file.utils import load_file, save_file
from jet.llm.ollama.base import Ollama, OllamaEmbedding
from jet.logger import logger
from jet.token.token_utils import filter_texts, split_docs
from jet.transformers.formatters import format_json
from jet.utils.markdown import extract_json_block_content
from jet.wordnet.sentence import split_sentences
from tqdm import tqdm

header_texts = [
    "# Top 30 Best Rom-Com Anime Of All Time: The Ultimate Ranking",
    "### 30. Oh! My Goddess",
    "### 29. The Tatami Galaxy",
    "### 28. My Senpai is Annoying",
    "### 27. Net-juu no Susume",
    "### 26. His and Her Circumstances",
    "### 25. Arakawa Under the Bridge",
    "### 24. Maid Sama!",
    "### 23. Kamisama Kiss",
    "### 22. High Score Girl",
    "### 21. Kimi ni Todoke: From Me to You",
    "### 20. My Little Monster",
    "### 19. The Pet Girl of Sakurasou",
    "### 18. Monthly Girls' Nozaki-kun",
    "### 17. Ouran High School Host Club",
    "### 16. The Quintessential Quintuplets",
    "### 15. Tonikawa: Over the Moon For You",
    "### 14. Lovely Complex",
    "### 13. Working!!",
    "### 12. Tsurezure Children",
    "### 11. School Rumble",
    "### 10. Nisekoi: False Love",
    "### 9. Saekano: How to Raise a Boring Girlfriend",
    "### 8. Wotakoi: Love is Hard for Otaku",
    "### 7. My Love Story!!",
    "### 6. Horimiya",
    "### 5. My Teen Romantic Comedy SNAFU",
    "### 4. Toradora!",
    "### 3. Teasing Master Takagi-san",
    "### 2. Ikkoku House",
    "### 1. Kaguya-sama: Love is War",
    "### R. Romero",
    "### Keep Browsing",
    "### Related Posts",
    "#### Browse Fandoms",
]

prompt_template = """
--- Documents ---
{headers}
--- End of Documents ---

Instruction:
{instruction}
Query: {query}
Answer:
""".strip()

if __name__ == "__main__":
    # llm_model = "gemma3:4b"
    llm_model = "mistral"
    embed_models = [
        "paraphrase-multilingual",
        # "mxbai-embed-large",
    ]
    embed_model = embed_models[0]

    sub_chunk_size = 128
    sub_chunk_overlap = 40

    instruction = "Given the provided documents, select ones that are relevant to the query in JSON format.\nReturn only the list of documents surrounded by ```json.\nOutput sample:\n```json\n[\n{\"doc\": 20,\n\"feedback\": \"Sample feedback\"}]\n```"
    query = "Top otome villainess anime today"

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_anime_scraper/myanimelist_net/scraped_html.html"
    output_dir = "generated"

    html: str = load_file(html_file)
    header_docs = get_docs_from_html(html)
    embed_model = OllamaEmbedding(model_name=embed_model)
    header_tokens: list[list[int]] = embed_model.encode(
        [d.text for d in header_docs])

    header_texts = []
    for doc_idx, doc in tqdm(enumerate(header_docs), total=len(header_docs)):
        sub_nodes = split_docs(
            doc, llm_model, tokens=header_tokens[doc_idx], chunk_size=sub_chunk_size, chunk_overlap=sub_chunk_overlap)
        parent_map = get_nodes_parent_mapping(sub_nodes, header_docs)

        sub_query = f"Query: {query}\n{doc.metadata["header"]}"
        reranked_sub_nodes = rerank_nodes(
            sub_query, sub_nodes, embed_models, parent_map)

        reranked_sub_text = "\n".join([n.text for n in reranked_sub_nodes[:3]])
        reranked_sub_text = reranked_sub_text.lstrip(
            doc.metadata["header"]).strip()
        header_texts.append(
            f"Document number: {doc.metadata["doc_index"] + 1}\n{doc.metadata["header"].lstrip("#").strip()}\n```text\n{reranked_sub_text}\n```"
        )

    header_texts = filter_texts(header_texts, llm_model)

    headers = "\n\n\n".join(header_texts)
    save_file(headers, os.path.join(
        output_dir, "filter_header_docs/top_nodes.md"))

    llm = Ollama(temperature=0.0, model=llm_model)

    message = prompt_template.format(
        headers=headers,
        query=query,
        instruction=instruction,
    )

    response = llm.chat(message, model=llm_model)
    json_result = extract_json_block_content(str(response))
    result = json.loads(json_result)

    logger.success(format_json(result))
