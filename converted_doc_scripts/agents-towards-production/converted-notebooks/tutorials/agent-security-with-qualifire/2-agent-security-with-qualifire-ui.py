from dotenv import load_dotenv
from jet.logger import CustomLogger
import PyPDF2
import io
import openai
import os
import qualifire
import shutil
import streamlit as st


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agent-security-with-qualifire--2-agent-security-with-qualifire-ui)

# Agent Guardrails with Qualifire 🔥

This notebook walks you through integrating Qualifire to add guardrails to an AI agent. We will build a simple chatbot using MLX's GPT-4.1 and Streamlit, and then implement protections against prompt injections, unsafe content, hallucinations, and policy breaches using both the Gateway and SDK.

## Overview

As AI agents become more sophisticated and widely deployed, ensuring their safe and controlled operation becomes critical. Without proper guardrails, AI agents can be vulnerable to various risks including prompt injections, unsafe content generation, hallucinations, and policy violations.

## Benefits of Using Guardrails

- **Enhanced Security**: Protect against prompt injection attacks and unauthorized access
- **Content Safety**: Filter and prevent generation of harmful or inappropriate content
- **Quality Control**: Reduce hallucinations and ensure factual responses
- **Policy Compliance**: Enforce organizational policies and usage guidelines
- **Real-time Protection**: Implement guardrails that work during runtime

## Key Methods

1. **Gateway Protection**: Implement API-level security controls
2. **Content Filtering**: Set up content moderation and safety checks
3. **Hallucination Prevention**: Add fact-checking and verification
4. **Policy Enforcement**: Configure and apply usage policies

ℹ️  If you're interested in seeing how the demo app is created visit the [streamlit tutorial](../agent-with-streamlit-ui/building-chatbot-notebook.ipynb) before proceeding with this tutorial.

<img src="./assets/freddie-shield.png" width="200px" alt="Qualifire Shield Logo">

## 1. Setup and Requirements
"""
logger.info("# Agent Guardrails with Qualifire 🔥")

# !pip install -q -r requirements.txt

"""
### 1.2. Sign up for Qualifire and Get API Key

ℹ️ If you've already created an API key, you can skip this step.

Before proceeding, make sure you have a Qualifire account and an API key.

1. Sign up at [https://app.qualifire.ai](https://app.qualifire.ai?utm=agents-towards-production).
2. complete the onboarding and create your API key.

<img src="./assets/protection-rules-evaluation.png" alt="Protection Rules Evaluation">
<img src="./assets/protection-rules-actions.png" alt="Protection Rules Actions">

3. once you see the "waiting for logs" screen you can proceed with the tutorial.

<img src="./assets/wait-for-logs.png" alt="Waiting for logs">

## 2. patching the streamlit app

## Setup: MLX API Key

To use MLX's API, you need to provide your API key so that the library can authenticate. There are a couple of ways to do this:

# 1. **Option 1 (Recommended)**: Set the API key as an environment variable on your system (e.g., `OPENAI_API_KEY`). This keeps the key out of your code.
#    - On Linux/Mac: `export OPENAI_API_KEY='your_key_here'` in your terminal
#    - On Windows: `set OPENAI_API_KEY="your_key_here"` in the Command Prompt

2. **Option 2**: Directly assign the API key in your code (quick for testing, but be careful not to expose your key if you share your code)

In this tutorial, we'll assume you saved your key as an environment variable for safety. It's a best practice to avoid hard-coding secrets.

ℹ️ You can use any LLM you'd like. For this tutorial, we'll use MLX's GPT-4.1. If you want to read the specific configurations for each LLM, check out the [documentation](https://docs.qualifire.ai?utm=agents-towards-production).

### 2.1 hooking Qualifire as a gateway
"""
logger.info("### 1.2. Sign up for Qualifire and Get API Key")


load_dotenv()  # Load environment variables from .env

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QUALIFIRE_API_KEY = os.getenv("QUALIFIRE_API_KEY")
qualifire.init(
    api_key=QUALIFIRE_API_KEY,
)








# if not OPENAI_API_KEY or not QUALIFIRE_API_KEY:
#     raise ValueError("OPENAI_API_KEY and QUALIFIRE_API_KEY must be set")




client = openai.MLX(
#     api_key=OPENAI_API_KEY,
    base_url="https://proxy.qualifire.ai/api/providers/openai",
    default_headers={
        "X-Qualifire-Api-Key": QUALIFIRE_API_KEY,
    },
)




...

"""
# 3. Creating a new protection rule

In Qualifire a protection rule is way to define a policy that will be applied to LLM calls, Applying guardrails and allowing you to set up an escape hatch for LLMs that are not behaving as expected.

1. go to the protection rules [tab](https://app.qualifire.ai/rules?utm=agents-towards-production)
2. name your rule and click next
3. select you evaluation and choose if it should run on the input or output (for now just the input)
4. Choose what action to take when the evaluation fails. In this example add a default response of "I can't do that"
5. click on create

#### now that you have a working agent you can interact with, let's try to bypass the guardrails

1. you can use [this](https://github.com/drorIvry/L1B3RT45/blob/main/OPENAI.mkd) for an initial set of jailbreaks and prompt injections. 
⚠️ this is a reference, use at your own risk

Try toggling the Qualifire jailbreak protection rule  on and off to see how it affects the LLM response.
"""
logger.info("# 3. Creating a new protection rule")

# !streamlit run app.py

"""
# <img src="./assets/prompt-injections-demo.gif">

# 3. Conclusion

Thank you for completing this tutorial! We hope it has been helpful in understanding how to use Qualifire to enhance the observability of your agents.

In this tutorial, we learned how to:

- Initialize Qualifire in your Python application with a single line of code.
- Run an agent, with Qualifire automatically capturing observability data via OpenTelemetry in the background.
- Setting up Qualifire Guardrails to protect LLM calls.

Using Qualifire provides deep unparalleled control, visibility and protection over your AI.


### Thank you for completing the tutorial! 🙏
we'd like to offer you 1 free month of the Pro plan to help you get started with Qualifire. use code `NIR1MONTH` at checkout

For more details visit [https://qualifire.ai](https://qualifire.ai?utm=agents-towards-production).
"""
logger.info("# <img src="./assets/prompt-injections-demo.gif">")

logger.info("\n\n[DONE]", bright=True)