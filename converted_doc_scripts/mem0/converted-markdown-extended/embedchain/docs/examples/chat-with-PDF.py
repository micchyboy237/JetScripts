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
### Embedchain Chat with PDF App

You can easily create and deploy your own `chat-pdf` App using Embedchain.

Here are few simple steps for you to create and deploy your app:

1. Fork the embedchain repo from [Github](https://github.com/embedchain/embedchain).

<Note>
If you run into problems with forking, please refer to [github docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) for forking a repo.
</Note>

2. Navigate to `chat-pdf` example app from your forked repo:
"""
logger.info("### Embedchain Chat with PDF App")

cd <your_fork_repo>/examples/chat-pdf

"""
3. Run your app in development environment with simple commands
"""
logger.info("3. Run your app in development environment with simple commands")

pip install -r requirements.txt
ec dev

"""
Feel free to improve our simple `chat-pdf` streamlit app and create pull request to showcase your app [here](https://docs.embedchain.ai/examples/showcase)

4. You can easily deploy your app using Streamlit interface

Connect your Github account with Streamlit and refer this [guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app) to deploy your app.

You can also use the deploy button from your streamlit website you see when running `ec dev` command.
"""
logger.info("Feel free to improve our simple `chat-pdf` streamlit app and create pull request to showcase your app [here](https://docs.embedchain.ai/examples/showcase)")

logger.info("\n\n[DONE]", bright=True)