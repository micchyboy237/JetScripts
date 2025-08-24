from dotenv import load_dotenv
from jet.llm.mlx.base_langchain import ChatMLX
from jet.logger import CustomLogger
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from openai import MLX
from pathlib import Path
import json
import openai
import os
import os, json, time
import pandas as pd
import pathlib
import random
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--fine-tunning-agents--fine-tuning-agents-guide)

# Enhancing AI Agents Through Fine-Tuning Guide
## Overview
This tutorial demonstrates how to leverage supervised fine-tuning (SFT) to create more capable and efficient AI agents. While we'll use a banking example for demonstration, the techniques and principles covered here apply to any scenario where you need to enhance an AI agent's capabilities through fine-tuning.

## What is Fine-Tuning?

Fine-tuning is the process of taking a pre-trained AI model—one that has already learned general language patterns from vast amounts of data—and further training it on a smaller, specialized dataset. 

This additional training helps the model adapt to specific domains, tasks, or communication styles, making it more effective and reliable for targeted applications. Fine-tuning allows you to customize the model’s behavior, vocabulary, and output format to better suit your unique needs.

## Why Fine-Tune AI Agents?
Fine-tuning offers several powerful advantages for AI agent development:

1. **Domain Expertise**: 
   - Internalizes domain-specific regulations and compliance requirements

   - Reduces hallucinations in specialized contexts

2. **Consistent Style & Tone**:
   - Learns to maintain your brand voice consistently

   - Provides uniform responses across all interactions

3. **Structured Outputs**:
   - Reliably generates specific formats (JSON, SQL, Markdown)

4. **Reduced Prompt Engineering**:
   - Eliminates need for lengthy system prompts

   - Significantly reduces token usage and costs

5. **Enhanced Efficiency**:
   - Smaller tuned models can replace larger base models

   - Faster response times for specialized tasks


### Who Should Consider Fine-Tuning?
- Teams building specialized AI agents for specific domains

- Applications where response speed and cost matter

- Systems handling complex domain-specific tasks

## Method Overview
1. **Data Preparation**: Creating high-quality training datasets that capture desired behaviors

2. **Supervised Fine-Tuning**: Adapting foundation models to specific requirements

3. **Integration**: Building efficient agent workflows with fine-tuned models

4. **Evaluation**: Measuring performance improvements across key metrics


This approach provides a robust framework for enhancing AI agents through fine-tuning, enabling them to perform specialized tasks more effectively while maintaining efficiency and reliability.

## Conclusion
This tutorial shows how to leverage fine-tuning to create more capable AI agents. Whether you need domain expertise, consistent formatting, or improved efficiency, fine-tuning can help your agents perform better while reducing costs and complexity. The techniques demonstrated here can be applied across any industry or use case where specialized agent behavior is desired.

Written by [Zohar Mosseri](https://www.linkedin.com/in/zohar-mosseri/)

📖 **For more background on fine-tuning AI models and their applications, check out our detailed blog post:** [Fine-tuning AI models](https://diamantai.substack.com/p/fine-tuning-ai-models-how-they-evolve)

*DiamantAI is a top 0.1% newsletter for AI with over 25,000 subscribers, focusing on AI techniques, breakthroughs, and tutorials.*

---

## Example Use Case: Banking Assistant
To demonstrate these concepts, we'll create a banking customer support assistant. This example will show how to:
- Structure training data for domain-specific knowledge
- Fine-tune a model for specialized tasks
- Integrate the fine-tuned model into a practical application

## 1. Setup and Imports

Install and import necessary libraries
"""
logger.info("# Enhancing AI Agents Through Fine-Tuning Guide")

# %pip install --upgrade openai langchain langgraph==0.3.1 tiktoken pandas scikit-learn


load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

client = MLX()

"""
## 2. Prepare Training & Validation Files for Fine-tuning

MLX expects **UTF‑8 encoded JSONL** files for fine-tuning, with one JSON object per line.
Each example should follow the chat format with system, user, and assistant messages.

### Understanding the Training Data Format
The quality of your fine-tuned model depends heavily on your training data. Here's what you need to know:

1. **File Format**: JSONL (JSON Lines) with UTF-8 encoding

2. **Message Structure**: Each example must contain a sequence of messages with roles and content

3. **Required Fields**: Each message needs `role` and `content` fields

4. **Valid Roles**: `system`, `user`, and `assistant`

### Example Format
```jsonl
{
  "messages": [
    {"role": "system", "content": "You are a helpful domain expert assistant."},
    {"role": "user", "content": "What is the policy on this matter?"},
    {"role": "assistant", "content": "According to our policies..."}
  ]
}
```

### Creating Training Data: A Banking Example
To demonstrate these principles, let's create a training dataset using a banking knowledge base. This example can be adapted for any domain-specific knowledge:
"""
logger.info("## 2. Prepare Training & Validation Files for Fine-tuning")



DATA_DIR = pathlib.Path("bank_finetune_data")
DATA_DIR.mkdir(exist_ok=True)


kb_docs = [
    {
        "title": "Account Types",
        "content": {
            "checking": "We offer three types of checking accounts: Basic (no minimum balance), Premium ($2,500 minimum, no fees), and Student (no fees with valid student ID).",
            "savings": "Our savings accounts include Regular (0.5% APY), High-Yield (1.5% APY with $10,000 minimum), and Goal-Based savings with customizable targets.",
            "business": "Business accounts feature unlimited transactions, merchant services integration, and dedicated support. Available in Standard and Premium tiers."
        }
    },
]

def generate_diverse_examples(kb_doc):
    """
    Generate multiple training examples with diverse phrasings and scenarios.
    Following best practices for high-quality training data.
    """
    examples = []
    content = kb_doc["content"]

    for subtopic, details in content.items():
        questions = [
            f"Can you explain {subtopic}?",
            f"What should I know about {subtopic}?",
            f"Tell me about your {subtopic}",
            f"How does {subtopic} work?",
            f"I need information regarding {subtopic}",
            f"What are the details of {subtopic}?"
        ]

        responses = [
            f"Here's what you need to know about {subtopic}: {details}",
            f"Regarding {subtopic}: {details} Let me know if you need any clarification.",
            f"I'll explain {subtopic}. {details} Is there anything specific you'd like to know more about?",
            f"{details} This information about {subtopic} is current as of today. Please check our website for any updates."
        ]

        system_messages = [
            "You are a knowledgeable banking assistant focused on providing accurate, compliant information.",
            "You are a helpful financial services expert committed to clear, precise communication.",
            "You are a banking specialist dedicated to providing detailed, accurate responses."
        ]

        for system_msg in system_messages:
            for question in questions:
                for response in responses:
                    examples.append({
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": response}
                        ]
                    })

    return examples

examples = []
for doc in kb_docs:
    examples.extend(generate_diverse_examples(doc))

random.shuffle(examples)

logger.debug(f"Generated {len(examples)} diverse training examples")

"""
### Data Quality Best Practices for Fine-Tuning

The success of your fine-tuned model depends on well-structured training data. Key principles:

1. **Data Requirements**
   - At least 10 examples per behavior
   - Balance types and reserve 10–20% for validation

2. **Quality Guidelines**
   - Clear intent and response in each example
   - Vary phrasing, keep formatting and style consistent
   - Reflect real-world usage

3. **Common Pitfalls**
   - Avoid sensitive info, inconsistent styles, duplicates, and overfitting

Remember: Quality over quantity—a small set of high-quality examples outperforms a large set of poor ones.

### Split into training and validation sets

A crucial step in fine-tuning is dividing your dataset into training and validation sets. The training set teaches the model, while the validation set helps measure its performance on unseen data. We'll use a 80/20 split - a common ratio that provides enough validation data while maximizing training examples.
"""
logger.info("### Data Quality Best Practices for Fine-Tuning")

random.shuffle(examples)

split_index = int(0.8 * len(examples))

train_examples, validation_examples = examples[:split_index], examples[split_index:]

train_file = DATA_DIR / "train.jsonl"
validation_file = DATA_DIR / "validation.jsonl"

with open(train_file, "w", encoding="utf-8") as f:
    for example in train_examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

with open(validation_file, "w", encoding="utf-8") as f:
    for example in validation_examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

logger.debug(f"✍️ Created {len(train_examples)} training examples and {len(validation_examples)} validation examples")

df = pd.DataFrame([
    {
        "system": ex["messages"][0]["content"],
        "user": ex["messages"][1]["content"],
        "assistant": ex["messages"][2]["content"]
    }
    for ex in train_examples[:3]
])
df

"""
### Validating the JSONL format

Before uploading, let's validate our files to ensure they meet MLX's requirements:
"""
logger.info("### Validating the JSONL format")

def validate_jsonl(file_path):
    """Validate that the file is proper JSONL format for fine-tuning."""
    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line_count += 1
            try:
                data = json.loads(line)
                if 'messages' not in data:
                    return False, f"Line {line_count} missing 'messages' field"
                for msg in data['messages']:
                    if 'role' not in msg or 'content' not in msg:
                        return False, f"Line {line_count} has message missing 'role' or 'content'"
                    if msg['role'] not in ['system', 'user', 'assistant']:
                        return False, f"Line {line_count} has invalid role: {msg['role']}"
            except json.JSONDecodeError:
                return False, f"Line {line_count} is not valid JSON"
    return True, f"Validated {line_count} examples"

train_valid, train_msg = validate_jsonl(train_file)
val_valid, val_msg = validate_jsonl(validation_file)

logger.debug(f"Training file: {'✅' if train_valid else '❌'} {train_msg}")
logger.debug(f"Validation file: {'✅' if val_valid else '❌'} {val_msg}")

"""
## 3. Upload your JSONL files to MLX and Launch Fine-tuning

We'll upload our files programmatically using the MLX Python client:
"""
logger.info("## 3. Upload your JSONL files to MLX and Launch Fine-tuning")


# client = MLX(api_key=os.environ["OPENAI_API_KEY"])

def upload(path):
    """Upload a file to MLX for fine-tuning"""
    with open(path, "rb") as file:
        resp = client.files.create(
            file=file,
            purpose="fine-tune"  # Specify the file will be used for fine-tuning
        )
    logger.debug(f"Uploaded {Path(path).name} → {resp.id}")
    return resp.id

train_file_id = upload(train_file)
validation_file_id = upload(validation_file)

"""
You can also upload them through the UI: 

1. Navigate to the dashboard > [fine-tuning](https://platform.openai.com/finetune).

2. Click + Create.

3. Under Training data, upload your JSONL files.

## 4.  Launch the fine‑tuning job
### Create fine-tuning job with the base model and the JSONL files
"""
logger.info("## 4.  Launch the fine‑tuning job")

job = client.fine_tuning.jobs.create(
    training_file=train_file_id,
    validation_file=validation_file_id,
    model="qwen3-1.7b-4bit-mini-2024-07-18",  # base model
    suffix="banking-support"  # custom suffix for the fine-tuned model
)
logger.debug(f"Fine-tuning job created with ID: {job.id}")
logger.debug(f"Job status: {job.status}")
logger.debug(f"Monitor progress at: https://platform.openai.com/finetune")

"""
## 5.  Monitor job status
"""
logger.info("## 5.  Monitor job status")


while True:
    status = client.fine_tuning.jobs.retrieve(job.id).status
    logger.debug("Status:", status)
    if status in ("succeeded", "failed", "cancelled"):
        break      # Exit loop if job reaches a terminal state
    time.sleep(30)

fine_tuned_model = openai.fine_tuning.jobs.retrieve(job.id).fine_tuned_model  # e.g. 'ft:gpt‑4o-mini:banking-support:2025-05-12-10-30-02'
logger.debug("✅ Fine‑tuned model:", fine_tuned_model)

"""
### Understanding Fine-Tuning Metrics

You can watch progress and loss curves in the [fine-tuning](https://platform.openai.com/finetune) tab of the MLX dashboard. The fine-tuning process typically takes a few minutes to several hours depending on dataset size and model complexity.


When fine-tuning, you'll see two key graphs:

#### Loss Curves
The loss curves show how well the model is learning over time:

- **Training Loss** (🟢 Green): Model's error on training data (should decrease)

- **Validation Loss** (🟣 Purple): Error on unseen data (should follow training loss)

- If validation loss increases while training loss decreases = overfitting

#### Accuracy Curves
These show the model's prediction accuracy over time:

- **Training Accuracy** (🟢 Green): Should steadily increase

- **Validation Accuracy** (🟣 Purple): Should follow training curve
- If they diverge = potential overfitting

#### What to Look For
✅ **Good Signs**
- Both metrics improving steadily

- Training and validation curves following similar trends

⚠️ **Warning Signs**
- Validation loss increasing while training loss decreases

- Large gaps between training and validation metrics

- Erratic or unstable curves

![Fine-tuning metrics showing loss and accuracy curves](https://i.ibb.co/hx55yKKH/loss-accuracy-curve.png)

## 6. Implementing Your Fine-Tuned Model

This section demonstrates how to integrate your fine-tuned model into a production environment using LangGraph. You'll learn how to:
- Connect your specialized model to existing tools and systems

- Structure the interaction flow

- Handle domain-specific queries effectively
"""
logger.info("### Understanding Fine-Tuning Metrics")



@tool
def account_lookup(account_id: str) -> str:
    """Look up account information by account ID (requires authentication)."""
    return f"Account {account_id} information is available after authentication."

@tool
def transfer_payment(source_account: str, destination_account: str, amount: float) -> str:
    """Transfer payment from one account to another."""
    return f"Payment of {amount} from {source_account} to {destination_account} has been initiated."


domain_expert = ChatMLX(model=fine_tuned_model, temperature=0.0)

agent = create_react_agent(
    model=domain_expert,  # Use our fine-tuned model
    tools=[account_lookup, transfer_payment]
    )

test_queries = [
    "What account types do you offer?",
    "What are the fees for wire transfers?",
    "I need information about overdraft protection"
]

for query in test_queries:
    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    logger.debug("\nQuery:", query)
    logger.debug("\nResponse:")
    for msg in reversed(response["messages"]):
        if hasattr(msg, "content") and msg.content and msg.__class__.__name__ == "AIMessage":
            logger.debug(msg.content)
            break

"""
*Tip: if you later need multiple specialised models, simply point each LangGraph node to a different fine tuned model.*

## 7. Benefits of Fine-Tuning: A Comparative Analysis

Let's compare how a fine-tuned model performs against a base LLM in two key areas:

### 1. Domain Expertise & Security
**Query:** "I want to transfer $50,000 to account 1234567"

**Base LLM:**
> For your safety, please make sure you are using a secure and verified banking platform to perform this action. If you’d like, I can draft a sample confirmation message or explain the common steps to transfer funds through your bank or app. Would you like me to do that?

**Fine-Tuned Model:**
> For your security, I'll need to:
> 1. Verify your identity through our authentication process
> 2. Confirm the destination account details
> 3. Check if this transfer requires additional fraud prevention measures (required for transfers over $10,000)
> 
> Please provide your account ID to begin the secure verification process.

### 2. Structured Responses & Product Knowledge
**Query:** "What savings accounts do you offer?"

**Base LLM:**
> Banks typically offer various types of savings accounts with different interest rates and minimum balance requirements. You should check with your specific bank for their offerings.

**Fine-Tuned Model:**
```json
{
    "savings_products": {
        "regular_savings": {
            "rate": "0.5% APY",
            "min_balance": "$0",
            "features": ["Mobile banking", "24/7 support"]
        },
        "high_yield_savings": {
            "rate": "1.5% APY",
            "min_balance": "$10,000",
            "features": ["Premium interest rate", "Priority support"]
        }
    }
}
```

**Takeaway:**  

These improvements would be difficult to achieve through prompt engineering alone, showing the value of fine-tuning for creating specialized AI agents.

## 8.  Best Practices for Fine-Tuning

| Topic                   | Recommendation                                                                                   |
|-------------------------|--------------------------------------------------------------------------------------------------|
| **Train/Validation Split** | Always reserve at least 10% of your data (ideally ≥40 examples) for validation. This helps you monitor overfitting and ensures your model generalizes well to new queries. |
| **Early Stopping**         | Enabled by default. Training will automatically stop if the validation loss stops improving, preventing overfitting and saving compute costs. |
| **Example Truncation**     | Any sample longer than the model’s context window is auto‑truncated. Keep messages concise and relevant for best results. |
| **Evaluation Metrics**     | The MLX dashboard shows token‑level accuracy. For deeper analysis, consider running BLEU or exact-match metrics on your own test set. |

## 9. Next Steps

- Automate Data Refresh: 
  Schedule regular (e.g., nightly) re-training with new domain data to keep the model up-to-date with evolving knowledge and requirements.

- Add Compliance & Audit Nodes:
  Integrate additional LangGraph nodes (potentially with their own fine-tuned models) to automatically check responses for compliance and flag sensitive topics.

- Monitor agent performance and costs with tools like LangSmith, and continuously evaluate real user queries to iterate and improve your training data and workflow.
"""
logger.info("## 7. Benefits of Fine-Tuning: A Comparative Analysis")

logger.info("\n\n[DONE]", bright=True)