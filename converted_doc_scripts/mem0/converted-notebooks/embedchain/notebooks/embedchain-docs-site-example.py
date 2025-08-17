from IPython.display import Markdown
from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


embedchain_docs_bot = App()

embedchain_docs_bot.add("docs_site", "https://docs.embedchain.ai/")

answer = embedchain_docs_bot.query("Write a flask API for embedchain bot")

markdown_answer = Markdown(answer)
display(markdown_answer)

logger.info("\n\n[DONE]", bright=True)