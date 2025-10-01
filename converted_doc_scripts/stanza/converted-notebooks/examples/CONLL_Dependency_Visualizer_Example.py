from jet.logger import logger
from stanza.utils.visualization.conll_deprel_visualization import conll_to_visual
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

conll_to_visual(en_file, "en", sent_count=2)
conll_to_visual(en_file, "en", sent_count=10)


jp_file = f"{os.path.dirname(__file__)}/data/japanese_test.conllu.txt"
conll_to_visual(jp_file, "ja")


ar_file = f"{os.path.dirname(__file__)}/data/arabic_test.conllu.txt"
conll_to_visual(ar_file, "ar")

logger.info("\n\n[DONE]", bright=True)