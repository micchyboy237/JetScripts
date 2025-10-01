from jet.adapters.stanza.conll_deprel_visualization import conll_to_visual
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


en_file = f"{os.path.dirname(__file__)}/data/en_test.conllu.txt"

html_strings = conll_to_visual(en_file, "en", sent_count=2)
for i, html in enumerate(html_strings):
  save_file(html, f"{OUTPUT_DIR}/en/html_sent_count_2_{i + 1}.html")
html_strings = conll_to_visual(en_file, "en", sent_count=10)
for i, html in enumerate(html_strings):
  save_file(html, f"{OUTPUT_DIR}/en/html_sent_count_10_{i + 1}.html")


jp_file = f"{os.path.dirname(__file__)}/data/japanese_test.conllu.txt"
html_strings = conll_to_visual(jp_file, "ja")
for i, html in enumerate(html_strings):
  save_file(html, f"{OUTPUT_DIR}/ja/html_japanese_test_{i + 1}.html")


ar_file = f"{os.path.dirname(__file__)}/data/arabic_test.conllu.txt"
html_strings = conll_to_visual(ar_file, "ar")
for i, html in enumerate(html_strings):
  save_file(html, f"{OUTPUT_DIR}/ar/html_arabic_test_{i + 1}.html")

logger.info("\n\n[DONE]", bright=True)