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
title: 'Overview'
description: 'Deploy your RAG application to production'
---

After successfully setting up and testing your RAG app locally, the next step is to deploy it to a hosting service to make it accessible to a wider audience. Embedchain provides integration with different cloud providers so that you can seamlessly deploy your RAG applications to production without having to worry about going through the cloud provider instructions. Embedchain does all the heavy lifting for you.

<CardGroup cols={4}>
  <Card title="Fly.io" href="/deployment/fly_io"></Card>
  <Card title="Modal.com" href="/deployment/modal_com"></Card>
  <Card title="Render.com" href="/deployment/render_com"></Card>
  <Card title="Railway.app" href="/deployment/railway"></Card>
  <Card title="Streamlit.io" href="/deployment/streamlit_io"></Card>
  <Card title="Gradio.app" href="/deployment/gradio_app"></Card>
  <Card title="Huggingface.co" href="/deployment/huggingface_spaces"></Card>
</CardGroup>

## Seeking help?

If you run into issues with deployment, please feel free to reach out to us via any of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Seeking help?")

logger.info("\n\n[DONE]", bright=True)