from IPython.display import HTML
from dotenv import load_dotenv
from jet.logger import logger
import ollama
import os
import shutil
import streamlit as st
import streamlit as st  # Import Streamlit for the UI


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
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-with-streamlit-ui--building-chatbot-notebook)

# Building a Chatbot AI Agent with Ollama and Streamlit

## Introduction

Have you ever wanted to create your own chatbot like ChatGPT? In this tutorial, we will build a simple AI chatbot from scratch using Ollama's API (which provides powerful language models like GPT-3.5) and Streamlit (a Python library for creating web apps).

By the end, you'll have a beginner-friendly chatbot that runs in your browser, with a chat interface for conversations and a file uploader for sharing files with the bot. We'll go step-by-step, explaining every part of the code so even absolute beginners can follow along.

**What we'll build:**
- A web application where you can type messages to an AI agent and receive responses
- The interface will resemble a chat window (like messaging apps)
- An option to upload a file (for example, a text file) that your chatbot could potentially use

The focus of this guide is on using Streamlit to create a clean and simple UI and integrating it with Ollama's API to handle the chatbot's intelligence. Let's get started!

## Requirements

Before coding, make sure you have the following:

- Python 3.x installed on your system
- An Ollama API key (required to access Ollama's language model). You can get one by creating an account on Ollama's website and generating an API key
- Basic knowledge of Python (functions, variables) – we will explain everything, but it helps to understand simple Python syntax
- Libraries: We'll use the `ollama` library to communicate with Ollama's API, and `streamlit` to build the web interface

## Installing the Required Libraries

We can install the necessary libraries using pip. Run the following in a terminal/command prompt:
"""
logger.info("# Building a Chatbot AI Agent with Ollama and Streamlit")

# !pip install ollama streamlit

"""
Alternatively, if you need additional libraries for file processing:
"""
logger.info("Alternatively, if you need additional libraries for file processing:")

# !pip install streamlit ollama requests PyPDF2

"""
## Setup: Ollama API Key

To use Ollama's API, you need to provide your API key so that the library can authenticate. There are a couple of ways to do this:

# 1. **Option 1 (Recommended)**: Set the API key as an environment variable on your system (e.g., `OPENAI_API_KEY`). This keeps the key out of your code.
#    - On Linux/Mac: `export OPENAI_API_KEY='your_key_here'` in your terminal
#    - On Windows: `set OPENAI_API_KEY="your_key_here"` in the Command Prompt

2. **Option 2**: Directly assign the API key in your code (quick for testing, but be careful not to expose your key if you share your code)

In this tutorial, we'll assume you saved your key as an environment variable for safety. It's a best practice to avoid hard-coding secrets.

to support option #1 we should install the python-dotenv package
"""
logger.info("## Setup: Ollama API Key")

# !pip install python-dotenv

"""
## Building the AI Agent (Ollama Integration)

First, let's write a small Python snippet to interact with Ollama's API. This will form the brain of our chatbot – it sends the user's message to Ollama and gets a response.
"""
logger.info("## Building the AI Agent (Ollama Integration)")


load_dotenv()  # Load environment variables from .env

# client = ollama.Ollama(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(user_prompt):
    """
    Sends the user prompt to Ollama and returns the AI's response.

    Parameters:
    -----------
    user_prompt : str
        The input message from the user.

    Returns:
    --------
    str
        The AI-generated response as plain text.
    """
    response = client.chat.completions.create(
        model="llama3.2",  # The AI model to use
        messages=[{"role": "user", "content": user_prompt}]  # The conversation context
    )
    message_text = response.choices[0].message.content
    return message_text  # Return the assistant's reply

test_reply = generate_response("Hello, how are you?")
logger.debug(test_reply)  # This should print an AI-generated response, e.g., "Hello! I'm doing well, how can I assist you?"

"""
Let's break down what's happening in `generate_response`:

- We call `ollama.Chat.completion.create(...)` with the model and a list of messages. The `messages` parameter expects a conversation history. We provide one message – the user's prompt – and specify its role as "user". You can also include a "system" message to prime the AI's behavior (for example, telling it to act as a friendly assistant), but we'll keep it simple for now.
- We chose the model "gpt-4o" which is the same model behind ChatGPT and is suitable for chat interactions.
- The Ollama API returns a response object that contains the AI's reply. The actual text of the reply is nested inside `response["choices"][0]["message"]["content"]`. We extract that and return it.
- The `logger.debug(test_reply)` line is just to verify that everything is working. It will print the AI's answer to "Hello, how are you?" in the console. When running as a Streamlit app, we won't use print; this is just a sanity check.

At this point, if you run this code in a regular Python environment (replacing the test prompt as needed and ensuring your API key is set), you should see a text response from the AI. This confirms our Ollama integration works. Now that the AI agent part is ready, let's build the web interface using Streamlit.

## Building the Streamlit UI

Streamlit makes it easy to create an interactive web interface with just Python code – no need to write HTML or JavaScript. We will create a chat-like interface where:

- The page has a title and a clean layout
- There's a sidebar with a file uploader widget (so users can upload a file, e.g., a text or PDF file)
- The main area displays the conversation (user and assistant messages)
- An input box at the bottom allows the user to type new messages

Let's go step by step.

### 1. Setting up the Streamlit app layout

We'll start by initializing the Streamlit app configuration, and adding a title at the top of the app.
"""
logger.info("## Building the Streamlit UI")


st.set_page_config(
    page_title="AI Chatbot",   # Title of the web page
    page_icon="🤖",           # An icon for the page (emoji of a robot)
    layout="wide"             # Use the full width of the page for a wide layout
)

st.title("🤖 AI Chatbot Assistant")
st.markdown("**Welcome!** Ask anything or upload a file for the bot to analyze.")

"""
Explanation:
- `st.set_page_config(...)` sets some global settings for the app. We give our app a title that will appear on the browser tab, an emoji icon (a robot face), and specify the layout as "wide" so the chat can use more horizontal space.
- `st.title("...")` displays a large heading at the top of the app. We included an emoji in the title as well just for style.
- `st.markdown("**Welcome!** ...")` adds a brief instruction or welcome message in bold. We use Markdown here to make the text bold.

### 2. Adding a File Uploader

Next, we'll add a file uploader component. This allows users to upload a file (like a text document) that the chatbot might use. We will put the uploader in the sidebar to keep the main interface clean.
"""
logger.info("### 2. Adding a File Uploader")

uploaded_file = st.sidebar.file_uploader(
    "Upload a file (optional):",  # Label for the uploader
    type=["txt", "pdf"]           # Limit file types to text or PDF for this example
)

if uploaded_file is not None:
    file_details = f"**{uploaded_file.name}** ({uploaded_file.size} bytes)"
    st.sidebar.write("Uploaded file:", file_details)

"""
Explanation:
- `st.sidebar.file_uploader(...)` creates a file uploader widget in the sidebar. The `type` parameter restricts the allowed file extensions (here .txt and .pdf for example purposes).
- We check `if uploaded_file is not None:` to see if the user has uploaded something. If a file is there, we retrieve some details:
  - `uploaded_file.name` gives the filename.
  - `uploaded_file.size` gives the file size in bytes.
- We display the file name and size in the sidebar using `st.sidebar.write`. We format the name in bold with Markdown.

Note: The sidebar in Streamlit can be shown or hidden by the user. If you run the app and don't see the uploader, look for a small arrow or the sidebar toggler on the page to expand it.

### 3. Maintaining Chat History with Session State

One important aspect of a chat interface is remembering the conversation. We want the app to show previous messages and replies, not just the latest one. Streamlit apps rerun the script from top to bottom on each user interaction, so without storing state, we'd lose the conversation history on each new message. To handle this, we use Streamlit's session state to store the messages.
"""
logger.info("### 3. Maintaining Chat History with Session State")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list to hold all messages (dicts with 'role' and 'content')

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm here to help. Feel free to ask me anything or upload a file for analysis."})

"""
Explanation:
- We use a key "messages" in `st.session_state` to hold the conversation. The first time the app runs, this key won't exist, so we initialize it to an empty list.
- The list `st.session_state.messages` will store messages as dictionaries like `{"role": "user", "content": "Hello"}` or `{"role": "assistant", "content": "Hi, how can I help?"}`. We will append to this list as the conversation grows.
- We included an optional step: if the messages list is empty (meaning no conversation yet), we append a greeting from the assistant. This way, when the user opens the app, they immediately see a welcome message from the chatbot.

Using session state in this way allows the app to remember past messages. Each time the script runs (for each new user input), it will retain `st.session_state.messages` from previous runs.

### 4. Displaying the Conversation

Now that we have a list of messages in state, we want to display them on the page in a chat-like format. Streamlit provides `st.chat_message` for this purpose, which is perfect for showing a chat bubble with either user or assistant style formatting.
"""
logger.info("### 4. Displaying the Conversation")

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("user"):
            st.markdown(msg["content"])

"""
Explanation:
- We loop through each message in the messages list.
- For each message, we check its "role". If the role is "assistant", we create a chat message container with `st.chat_message("assistant")`. If the role is "user", we use `st.chat_message("user")`.
- Using `with st.chat_message(<role>):` opens a container styled as a chat bubble for that role. Inside that with block, we output the message content. We use `st.markdown` to render the text.
- Streamlit automatically styles "assistant" messages differently from "user" messages (for example, different background color, and user messages might be right-aligned).

After this loop runs, the app will have rendered all previous messages in order. So the user can scroll up and see the conversation history.

### 5. Sending New Messages (Chat Input)

Finally, we need an input box for the user to type new messages. Streamlit's `st.chat_input()` provides a text input field fixed at the bottom of the page, which is perfect for chat apps. When the user enters a message and hits Enter, we will:

1. Capture that message
2. Add it to the session state history
3. Send it to the Ollama API (using our `generate_response` function from earlier)
4. Get the AI's reply and add that to the history

The app will then rerun and display the updated conversation (including the new messages) in the loop we wrote above.
"""
logger.info("### 5. Sending New Messages (Chat Input)")

user_message = st.chat_input("Type your message here...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):  # Show a spinner while waiting for the AI
            assistant_reply = generate_response(user_message)
            st.markdown(assistant_reply)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

"""
Explanation:
- `user_message = st.chat_input("Type your message here...")` creates a text input at the bottom. When the user submits a message, `st.chat_input` returns that message as a string.
- We check `if user_message:` which will be true if the user just sent a message.
- We then go through the steps:
  1. Append the user's message to the messages history list with role "user".
  2. Immediately display the user message in the chat. We do this so the user sees their message appear right away in the interface.
  3. To get the AI's response, we open an `st.chat_message("assistant")` context for the incoming reply. Within that, we use `st.spinner("Thinking...")` to show a loading spinner while the API call is in progress. We call `generate_response(user_message)` to get the AI's answer, then display it.
  4. We append the assistant's reply to the messages history list, with role "assistant".

## Complete Code

Here's the full code combining all the snippets above into a single script:
"""
logger.info("## Complete Code")


# ollama.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(user_prompt):
    response = ollama.ChatCompletion.create(
        model="llama3.2",
        messages=[{"role": "user", "content": user_prompt}]
    )
    message_text = response["choices"][0]["message"]["content"]
    return message_text

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot Assistant")
st.markdown("**Welcome!** Ask anything or upload a file for the bot to analyze.")

uploaded_file = st.sidebar.file_uploader("Upload a file (optional):", type=["txt", "pdf"])
if uploaded_file is not None:
    st.sidebar.write("Uploaded file:", f"**{uploaded_file.name}** ({uploaded_file.size} bytes)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm here to help. Feel free to ask me anything or upload a file."})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_msg := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_msg = generate_response(user_msg)
            st.markdown(assistant_msg)
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})

"""
## Running the App

Now that we have the code (for example, in a file named app.py), we can run the Streamlit app. In a terminal, make sure you are in the directory containing app.py and run:
"""
logger.info("## Running the App")



"""
This command will launch the Streamlit development server and open a web browser to the app (usually at http://localhost:8501). You should see your chatbot interface with the title and an initial greeting from the assistant.

Try it out: type a question into the chat box (for example, "What is the capital of France?") and hit Enter. You should see your message appear on the right, and after a moment, the AI's response will appear on the left (the app might show "Thinking..." while waiting for the response).

Note: Ensure your Ollama API key is valid and you have internet access when running the app, because the app needs to call Ollama's servers to get responses.

## Streamlit Chatbot Interface
# 
##### Here's what your chatbot interface should look like when running:
# 
# ![Streamlit Chatbot Interface](assets/streamlit_chatbot.png)
# 
##### This shows the chat interface with message history, input box at the bottom, and the file upload option in the sidebar.

## Streamlit Chatbot Demo
"""
logger.info("## Streamlit Chatbot Interface")


HTML("""
<video width="960" height="720" controls>
  <source src="assets/streamlit_chatbot_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
""")

"""
##### This video shows how users can interact with the chatbot in real-time, ask questions, and receive responses.

## Conclusion

Congratulations on building your first AI chatbot with Streamlit and Ollama! 🎉 In this tutorial, we covered:

- Installing and setting up the necessary libraries (streamlit and ollama)
- Obtaining and using an Ollama API key to access a GPT-4o model
- Writing a function to communicate with the Ollama API and get responses
- Using Streamlit to create a web interface, including a chat message display and input box, as well as a file uploader in the sidebar
- Maintaining state (conversation history) across interactions using st.session_state, enabling a multi-turn conversation
- Running the Streamlit app and interacting with the chatbot through your browser

This basic app provides a foundation that you can extend in many ways:

- **Improve the AI's context**: We kept the AI calls stateless (only sending the latest user prompt). You could send the whole conversation in the messages to ChatCompletion.create so the AI remembers previous questions.
- **Use the uploaded file**: Integrate the file content into the conversation. For example, if a PDF is uploaded, extract its text and prepend it as a system message.
- **UI enhancements**: Streamlit's chat elements support things like avatars and you can add more design elements (colors, sidebar info, etc.).
- **Deployment**: You can easily deploy this app to the web (for example, using Streamlit Cloud) and share your chatbot with friends or colleagues.

With relatively few lines of code, we created an interactive AI assistant! Feel free to experiment and build on this. Happy coding, and enjoy chatting with your new AI chatbot! 🚀


"""
logger.info("##### This video shows how users can interact with the chatbot in real-time, ask questions, and receive responses.")

logger.info("\n\n[DONE]", bright=True)