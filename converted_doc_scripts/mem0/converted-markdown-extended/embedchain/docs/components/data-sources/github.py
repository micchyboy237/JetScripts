from embedchain.chunkers.common_chunker import CommonChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.loaders.github import GithubLoader
from embedchain.pipeline import Pipeline as App
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
title: 📝 Github
---

1. Setup the Github loader by configuring the Github account with username and personal access token (PAT). Check out [this](https://docs.github.com/en/enterprise-server@3.6/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token) link to learn how to create a PAT.
"""
logger.info("title: 📝 Github")


loader = GithubLoader(
    config={
        "token":"ghp_xxxx"
        }
    )

"""
2. Once you setup the loader, you can create an app and load data using the above Github loader
"""
logger.info("2. Once you setup the loader, you can create an app and load data using the above Github loader")


# os.environ["OPENAI_API_KEY"] = "sk-xxxx"

app = App()

app.add("repo:embedchain/embedchain type:repo", data_type="github", loader=loader)

response = app.query("What is Embedchain?")

"""
The `add` function of the app will accept any valid github query with qualifiers. It only supports loading github code, repository, issues and pull-requests.
<Note>
You must provide qualifiers `type:` and `repo:` in the query. The `type:` qualifier can be a combination of `code`, `repo`, `pr`, `issue`, `branch`, `file`. The `repo:` qualifier must be a valid github repository name.
</Note>

<Card title="Valid queries" icon="lightbulb" iconType="duotone" color="#ca8b04">
    - `repo:embedchain/embedchain type:repo` - to load the repository
    - `repo:embedchain/embedchain type:branch name:feature_test` - to load the branch of the repository
    - `repo:embedchain/embedchain type:file path:README.md` - to load the specific file of the repository
    - `repo:embedchain/embedchain type:issue,pr` - to load the issues and pull-requests of the repository
    - `repo:embedchain/embedchain type:issue state:closed` - to load the closed issues of the repository
</Card>

3. We automatically create a chunker to chunk your GitHub data, however if you wish to provide your own chunker class. Here is how you can do that:
"""
logger.info("The `add` function of the app will accept any valid github query with qualifiers. It only supports loading github code, repository, issues and pull-requests.")


github_chunker_config = ChunkerConfig(chunk_size=2000, chunk_overlap=0, length_function=len)
github_chunker = CommonChunker(config=github_chunker_config)

app.add(load_query, data_type="github", loader=loader, chunker=github_chunker)

logger.info("\n\n[DONE]", bright=True)