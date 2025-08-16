from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# How to build and run the website

## Prerequisites

- dotnet 8.0 or later

## Build

Firstly, go to autogen/dotnet folder and run the following command to build the website:
"""
logger.info("# How to build and run the website")

dotnet tool restore
dotnet tool run docfx ../docs/dotnet/docfx.json --serve

"""
After the command is executed, you can open your browser and navigate to `http://localhost:8080` to view the website.
"""
logger.info("After the command is executed, you can open your browser and navigate to `http://localhost:8080` to view the website.")

logger.info("\n\n[DONE]", bright=True)