from embedchain import App
from embedchain.loaders.github import GithubLoader
from jet.logger import CustomLogger
from my_module import CustomChunker
from my_module import CustomLoader
import os
import shutil
import your_loader


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: '⚙️ Custom'
---

When we say "custom", we mean that you can customize the loader and chunker to your needs. This is done by passing a custom loader and chunker to the `add` method.
"""
logger.info("title: '⚙️ Custom'")


app = App()
loader = CustomLoader()
chunker = CustomChunker()

app.add("source", data_type="custom", loader=loader, chunker=chunker)

"""
<Note>
    The custom loader and chunker must be a class that inherits from the [`BaseLoader`](https://github.com/embedchain/embedchain/blob/main/embedchain/loaders/base_loader.py) and [`BaseChunker`](https://github.com/embedchain/embedchain/blob/main/embedchain/chunkers/base_chunker.py) classes respectively.
</Note>

<Note>
    If the `data_type` is not a valid data type, the `add` method will fallback to the `custom` data type and expect a custom loader and chunker to be passed by the user.
</Note>

Example:
"""
logger.info("The custom loader and chunker must be a class that inherits from the [`BaseLoader`](https://github.com/embedchain/embedchain/blob/main/embedchain/loaders/base_loader.py) and [`BaseChunker`](https://github.com/embedchain/embedchain/blob/main/embedchain/chunkers/base_chunker.py) classes respectively.")


app = App()

loader = GithubLoader(config={"token": "ghp_xxx"})

app.add("repo:embedchain/embedchain type:repo", data_type="github", loader=loader)

app.query("What is Embedchain?")

logger.info("\n\n[DONE]", bright=True)