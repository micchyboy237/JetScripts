from jet.code.markdown_utils import base_parse_markdown
from jet.code.markdown_utils._converters import convert_markdown_to_text
from jet.adapters.stanza.ner_visualization import visualize_strings as visualize_ner_strings
from jet.adapters.stanza.dependency_visualization import visualize_strings as visualize_dep_strings
from jet.adapters.stanza.semgrex_visualization import visualize_strings as visualize_sem_strings
from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

text = """Title: Headhunted to Another World: From Salaryman to Big Four!
Isekai
Fantasy
Comedy
Release Date: January 1, 2025
Japanese Title: Salaryman ga Isekai ni Ittara Shitennou ni Natta Hanashi Studio

Studio: Geek Toys, CompTown

Based On: Manga

Creator: Benigashira

Streaming Service(s): Crunchyroll
Powered by
Expand Collapse
Plenty of 2025 isekai anime will feature OP protagonists capable of brute-forcing their way through any and every encounter, so it is always refreshing when an MC comes along that relies on brain rather than brawn. A competent office worker who feels underappreciated, Uchimura is suddenly summoned to another world by a demonic ruler, who comes with quite an unusual offer: Join the crew as one of the Heavenly Kings. So, Uchimura starts a new career path that tasks him with tackling challenges using his expertise in discourse and sales.
Related"""
docs = [
    text,
]
# docs = load_sample_data(model="embeddinggemma", chunk_size=200, truncate=True)

if __name__ == "__main__":
    for i, md_content in enumerate(docs):
        save_file(md_content, f"{OUTPUT_DIR}/doc_{i + 1}/doc.md")

    doc_md_tokens = [base_parse_markdown(md_content, ignore_links=True) for md_content in docs]
    for i, md_tokens in enumerate(doc_md_tokens):
        save_file(md_tokens, f"{OUTPUT_DIR}/doc_{i + 1}/doc_md_tokens.json")

    doc_texts = [convert_markdown_to_text(md_content) for md_content in docs]
    for i, text in enumerate(doc_texts):
        save_file(text, f"{OUTPUT_DIR}/doc_{i + 1}/doc.txt")

    html_ner_strings = visualize_ner_strings(doc_texts, "en")
    for i, html in enumerate(html_ner_strings):
        save_file(html, f"{OUTPUT_DIR}/doc_{i + 1}/ner.html")
    
    html_dep_strings = visualize_dep_strings(doc_texts, "en")
    for i, html in enumerate(html_dep_strings):
        save_file(html, f"{OUTPUT_DIR}/doc_{i + 1}/dependency.html")

    queries = ["{pos:NN}=object <obl {}=action",
               "{cpos:NOUN}=thing <obj {cpos:VERB}=action"]
    html_sem_strings = visualize_sem_strings(doc_texts, queries, "en")
    for i, html in enumerate(html_sem_strings):
        save_file(html, f"{OUTPUT_DIR}/doc_{i + 1}/semgrex.html")
