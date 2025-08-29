from base import PDFNougatOCR
from google.colab import files
from jet.logger import CustomLogger
from pathlib import Path
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

# !pip install -qU nougat-ocr llama-index


upload = files.upload()


upload = files.upload()


reader = PDFNougatOCR()
pdf_path = Path("mathpaper.pdf")

docs = reader.load_data(pdf_path)

len(docs)

logger.info("\n\n[DONE]", bright=True)