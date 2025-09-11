from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.chat_message_histories import (
StreamlitChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import shutil
import streamlit as st


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Streamlit

>[Streamlit](https://docs.streamlit.io/) is an open-source Python library that makes it easy to create and share beautiful, 
custom web apps for machine learning and data science.

This notebook goes over how to store and use chat message history in a `Streamlit` app. `StreamlitChatMessageHistory` will store messages in
[Streamlit session state](https://docs.streamlit.io/library/api-reference/session-state)
at the specified `key=`. The default key is `"langchain_messages"`.

- Note, `StreamlitChatMessageHistory` only works when run in a Streamlit app.
- You may also be interested in [StreamlitCallbackHandler](/docs/integrations/callbacks/streamlit) for LangChain.
- For more on Streamlit check out their
[getting started documentation](https://docs.streamlit.io/library/get-started).

The integration lives in the `langchain-community` package, so we need to install that. We also need to install `streamlit`.

```
pip install -U langchain-community streamlit
```

You can see the [full app example running here](https://langchain-st-memory.streamlit.app/), and more examples in
[github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""
logger.info("# Streamlit")


history = StreamlitChatMessageHistory(key="chat_messages")

history.add_user_message("hi!")
history.add_ai_message("whats up?")

history.messages

"""
We can easily combine this message history class with [LCEL Runnables](/docs/how_to/message_history).

The history will be persisted across re-runs of the Streamlit app within a given user session. A given `StreamlitChatMessageHistory` will NOT be persisted or shared across user sessions.
"""
logger.info("We can easily combine this message history class with [LCEL Runnables](/docs/how_to/message_history).")

msgs = StreamlitChatMessageHistory(key="special_app_key")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOllama(model="llama3.2")

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)

"""
Conversational Streamlit apps will often re-draw each previous chat message on every re-run. This is easy to do by iterating through `StreamlitChatMessageHistory.messages`:
"""
logger.info("Conversational Streamlit apps will often re-draw each previous chat message on every re-run. This is easy to do by iterating through `StreamlitChatMessageHistory.messages`:")


for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.content)

"""
**[View the final app](https://langchain-st-memory.streamlit.app/).**
"""

logger.info("\n\n[DONE]", bright=True)