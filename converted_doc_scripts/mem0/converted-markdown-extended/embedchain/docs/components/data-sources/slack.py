from embedchain import App
from embedchain.chunkers.slack import SlackChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.loaders.slack import SlackLoader
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
title: '🤖 Slack'
---

## Pre-requisite
- Download required packages by running `pip install --upgrade "embedchain[slack]"`.
- Configure your slack bot token as environment variable `SLACK_USER_TOKEN`.
    - Find your user token on your [Slack Account](https://api.slack.com/authentication/token-types)
    - Make sure your slack user token includes [search](https://api.slack.com/scopes/search:read) scope.

## Example

### Get Started

This will automatically retrieve data from the workspace associated with the user's token.
"""
logger.info("## Pre-requisite")


os.environ["SLACK_USER_TOKEN"] = "xoxp-xxx"
app = App()

app.add("in:general", data_type="slack")

result = app.query("what are the messages in general channel?")

logger.debug(result)

"""
### Customize your SlackLoader
1. Setup the Slack loader by configuring the Slack Webclient.
"""
logger.info("### Customize your SlackLoader")


os.environ["SLACK_USER_TOKEN"] = "xoxp-*"

config = {
    'base_url': slack_app_url,
    'headers': web_headers,
    'team_id': slack_team_id,
}

loader = SlackLoader(config)

"""
NOTE: you can also pass the `config` with `base_url`, `headers`, `team_id` to setup your SlackLoader.

2. Once you setup the loader, you can create an app and load data using the above slack loader
"""
logger.info("NOTE: you can also pass the `config` with `base_url`, `headers`, `team_id` to setup your SlackLoader.")


app = App()

app.add("in:random", data_type="slack", loader=loader)
question = "Which bots are available in the slack workspace's random channel?"

"""
3. We automatically create a chunker to chunk your slack data, however if you wish to provide your own chunker class. Here is how you can do that:
"""
logger.info("3. We automatically create a chunker to chunk your slack data, however if you wish to provide your own chunker class. Here is how you can do that:")


slack_chunker_config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
slack_chunker = SlackChunker(config=slack_chunker_config)

app.add(slack_chunker, data_type="slack", loader=loader, chunker=slack_chunker)

logger.info("\n\n[DONE]", bright=True)