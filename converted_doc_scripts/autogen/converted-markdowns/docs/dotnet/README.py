from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
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