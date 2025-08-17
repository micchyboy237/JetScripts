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
---
title: 'Full Stack'
---

The Full Stack app example can be found [here](https://github.com/mem0ai/mem0/tree/main/embedchain/examples/full_stack).

This guide will help you setup the full stack app on your local machine.

### 🐳 Docker Setup

- Create a `docker-compose.yml` file and paste the following code in it.
"""
logger.info("### 🐳 Docker Setup")

version: "3.9"

services:
  backend:
    container_name: embedchain-backend
    restart: unless-stopped
    build:
      context: backend
      dockerfile: Dockerfile
    image: embedchain/backend
    ports:
      - "8000:8000"

  frontend:
    container_name: embedchain-frontend
    restart: unless-stopped
    build:
      context: frontend
      dockerfile: Dockerfile
    image: embedchain/frontend
    ports:
      - "3000:3000"
    depends_on:
      - "backend"

"""
- Run the following command,
"""

docker-compose up

"""
📝 Note: The build command might take a while to install all the packages depending on your system resources.

![Fullstack App](https://github.com/embedchain/embedchain/assets/73601258/c7c04bbb-9be7-4669-a6af-039e7e972a13)

### 🚀 Usage Instructions

- Go to [http://localhost:3000/](http://localhost:3000/) in your browser to view the dashboard.
- Add your `MLX API key` 🔑 in the Settings.
- Create a new bot and you'll be navigated to its page.
- Here you can add your data sources and then chat with the bot.

🎉 Happy Chatting! 🎉
"""
logger.info("### 🚀 Usage Instructions")

logger.info("\n\n[DONE]", bright=True)