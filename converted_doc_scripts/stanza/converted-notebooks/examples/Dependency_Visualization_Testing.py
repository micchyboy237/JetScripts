from jet.file.utils import save_file
from jet.logger import logger
from jet.adapters.stanza.dependency_visualization import visualize_strings
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


ar_strings = ['برلين ترفض حصول شركة اميركية على رخصة تصنيع دبابة "ليوبارد" الالمانية', "هل بإمكاني مساعدتك؟",
              "أراك في مابعد", "لحظة من فضلك"]
html_strings = visualize_strings(ar_strings, "ar")
for i, html in enumerate(html_strings):
  save_file(html, f"{OUTPUT_DIR}/ar/html_{i + 1}.html")


en_strings = ["This is a sentence.",
              "He is wearing a red shirt",
              "Barack Obama was born in Hawaii. He was elected President of the United States in 2008."]
html_strings = visualize_strings(en_strings, "en")
for i, html in enumerate(html_strings):
  save_file(html, f"{OUTPUT_DIR}/en/html_{i + 1}.html")


zh_strings = ["中国是一个很有意思的国家。"]
html_strings = visualize_strings(zh_strings, "zh")
for i, html in enumerate(html_strings):
  save_file(html, f"{OUTPUT_DIR}/zh/html_{i + 1}.html")

logger.info("\n\n[DONE]", bright=True)