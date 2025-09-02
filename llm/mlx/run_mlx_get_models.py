import os
import shutil
from jet.llm.mlx.generation import get_models
from jet.logger import CustomLogger
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

response = get_models()
save_file(response, f"{OUTPUT_DIR}/models.json")
