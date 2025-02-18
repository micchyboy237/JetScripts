from jet.executor.command import run_command
from jet.file import load_file
from jet.file.utils import save_file
from jet.logger import logger
import json

python_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/run_clean_jobs.py"
command = f"python {python_file}"

for line in run_command(command):
    if line.startswith("error: "):
        message = line[len("error: "):-2]
        logger.error(message)
    else:
        message = line[len("data: "):-2]
        logger.success(message)
