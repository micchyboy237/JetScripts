from jet.logger import logger
from stanza.utils.visualization.dependency_visualization import visualize_strings
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
visualize_strings(ar_strings, "ar")


en_strings = ["This is a sentence.",
              "He is wearing a red shirt",
              "Barack Obama was born in Hawaii. He was elected President of the United States in 2008."]
visualize_strings(en_strings, "en")


zh_strings = ["中国是一个很有意思的国家。"]
visualize_strings(zh_strings, "zh")

logger.info("\n\n[DONE]", bright=True)