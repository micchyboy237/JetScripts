from jet.adapters.stanza.semgrex_visualization import visualize_search_doc
from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil
import stanza


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


def main():
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

    doc = nlp("Banning opal removed artifact decks from the meta. Banning tennis resulted in players banning people.")
    queries = ["{pos:NN}=object <obl {}=action",
               "{cpos:NOUN}=thing <obj {cpos:VERB}=action"]
    html_strings = visualize_search_doc(doc, queries, "en")
    for i, html in enumerate(html_strings):
        save_file(html, f"{OUTPUT_DIR}/edited_html_{i + 1}.html")
    logger.debug(f"HTML Results: {len(html_strings)}")  # see the first sentence's deprel visualization HTML
    return


if __name__ == '__main__':
    main()

logger.info("\n\n[DONE]", bright=True)