from embedchain import App
from jet.logger import CustomLogger
import os
import shutil
import streamlit as st


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: '🚀 Streamlit'
description: 'Integrate with Streamlit to plug and play with any LLM'
---

In this example, we will learn how to use `mistralai/Mixtral-8x7B-Instruct-v0.1` and Embedchain together with Streamlit to build a simple RAG chatbot.

![Streamlit + Embedchain Demo](https://github.com/embedchain/embedchain/assets/73601258/052f7378-797c-41cf-ac81-f004d0d44dd1)

## Setup

Install Embedchain and Streamlit.
"""
logger.info("## Setup")

pip install embedchain streamlit

"""
<Tabs>
    <Tab title="app.py">
    ```python

    with st.sidebar:
        huggingface_access_token = st.text_input("Hugging face Token", key="chatbot_api_key", type="password")
        "[Get Hugging Face Access Token](https://huggingface.co/settings/tokens)"
        "[View the source code](https://github.com/embedchain/examples/mistral-streamlit)"


    st.title("💬 Chatbot")
    st.caption("🚀 An Embedchain app powered by Mistral!")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """
logger.info("import os")
            Hi! I'm a chatbot. I can answer questions and learn new things!\n
            Ask me anything and if you want me to learn something do `/add <source>`.\n
            I can learn mostly everything. :)
            """,
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything!"):
        if not st.session_state.chatbot_api_key:
            st.error("Please enter your Hugging Face Access Token")
            st.stop()

        os.environ["HUGGINGFACE_ACCESS_TOKEN"] = st.session_state.chatbot_api_key
        app = App.from_config(config_path="config.yaml")

        if prompt.startswith("/add"):
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
            prompt = prompt.replace("/add", "").strip()
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Adding to knowledge base...")
                app.add(prompt)
                message_placeholder.markdown(f"Added {prompt} to knowledge base!")
                st.session_state.messages.append({"role": "assistant", "content": f"Added {prompt} to knowledge base!"})
                st.stop()

        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            msg_placeholder.markdown("Thinking...")
            full_response = ""

            for response in app.chat(prompt):
                msg_placeholder.empty()
                full_response += response

            msg_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        ```
    </Tab>
    <Tab title="config.yaml">
    ```yaml
    app:
        config:
            name: 'mistral-streamlit-app'

    llm:
        provider: huggingface
        config:
            model: 'mistralai/Mixtral-8x7B-Instruct-v0.1'
            temperature: 0.1
            max_tokens: 250
            top_p: 0.1
            stream: true

    embedder:
        provider: huggingface
        config:
            model: 'sentence-transformers/all-mpnet-base-v2'
    ```
    </Tab>
</Tabs>

## To run it locally,
"""
logger.info("## To run it locally,")

streamlit run app.py

logger.info("\n\n[DONE]", bright=True)