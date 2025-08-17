from jet.logger import CustomLogger
from mem0 import MemoryClient
from openai import MLX
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
title: Memory-Guided Content Writing
---

This guide demonstrates how to leverage **Mem0** to streamline content writing by applying your unique writing style and preferences using persistent memory.

## Why Use Mem0?

Integrating Mem0 into your writing workflow helps you:

1. **Store persistent writing preferences** ensuring consistent tone, formatting, and structure.
2. **Automate content refinement** by retrieving preferences when rewriting or reviewing content.
3. **Scale your writing style** so it applies consistently across multiple documents or sessions.

## Setup
"""
logger.info("## Why Use Mem0?")


os.environ["MEM0_API_KEY"] = "your-mem0-api-key"
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


client = MemoryClient()
openai = MLX()

USER_ID = "content_writer"
RUN_ID = "smart_editing_session"

"""
## **Storing Your Writing Preferences in Mem0**
"""
logger.info("## **Storing Your Writing Preferences in Mem0**")

def store_writing_preferences():
    """Store your writing preferences in Mem0."""

    preferences = """My writing preferences:
1. Use headings and sub-headings for structure.
2. Keep paragraphs concise (8–10 sentences max).
3. Incorporate specific numbers and statistics.
4. Provide concrete examples.
5. Use bullet points for clarity.
6. Avoid jargon and buzzwords."""

    messages = [
        {"role": "user", "content": "Here are my writing style preferences."},
        {"role": "assistant", "content": preferences}
    ]

    response = client.add(
        messages,
        user_id=USER_ID,
        run_id=RUN_ID,
        metadata={"type": "preferences", "category": "writing_style"}
    )

    return response

"""
## **Editing Content Using Stored Preferences**
"""
logger.info("## **Editing Content Using Stored Preferences**")

def apply_writing_style(original_content):
    """Use preferences stored in Mem0 to guide content rewriting."""

    results = client.search(
        query="What are my writing style preferences?",
        version="v2",
        filters={
            "AND": [
                {
                    "user_id": USER_ID
                },
                {
                    "run_id": RUN_ID
                }
            ]
        },
    )

    if not results:
        logger.debug("No preferences found.")
        return None

    preferences = "\n".join(r["memory"] for r in results.get('results', []))

    system_prompt = f"""
You are a writing assistant.

Apply the following writing style preferences to improve the user's content:

Preferences:
{preferences}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Original Content:
    {original_content}"""}
    ]

    response = openai.chat.completions.create(
        model="llama-3.2-3b-instruct",
        messages=messages
    )
    clean_response = response.choices[0].message.content.strip()

    return clean_response

"""
## **Complete Workflow: Content Editing**
"""
logger.info("## **Complete Workflow: Content Editing**")

def content_writing_workflow(content):
    """Automated workflow for editing a document based on writing preferences."""

    store_writing_preferences()  # Ideally done once, or with a conditional check

    edited_content = apply_writing_style(content)

    if not edited_content:
        return "Failed to edit document."

    logger.debug("\n=== ORIGINAL DOCUMENT ===\n")
    logger.debug(content)

    logger.debug("\n=== EDITED DOCUMENT ===\n")
    logger.debug(edited_content)

    return edited_content

"""
## **Example Usage**
"""
logger.info("## **Example Usage**")

original_content = """Project Proposal

The following proposal outlines our strategy for the Q3 marketing campaign.
We believe this approach will significantly increase our market share.

Increase brand awareness
Boost sales by 15%
Expand our social media following

We plan to launch the campaign in July and continue through September.
"""

result = content_writing_workflow(original_content)

"""
## **Expected Output**

Your document will be transformed into a structured, well-formatted version based on your preferences.

### **Original Document**

Project Proposal
    
The following proposal outlines our strategy for the Q3 marketing campaign. 
We believe this approach will significantly increase our market share.

Increase brand awareness
Boost sales by 15%
Expand our social media following

We plan to launch the campaign in July and continue through September.

### **Edited Document**

# **Project Proposal**

## **Q3 Marketing Campaign Strategy**

This proposal outlines our strategy for the Q3 marketing campaign. We aim to significantly increase our market share with this approach.

### **Objectives**

- **Increase Brand Awareness**: Implement targeted advertising and community engagement to enhance visibility.
- **Boost Sales by 15%**: Increase sales by 15% compared to Q2 figures.
- **Expand Social Media Following**: Grow our social media audience by 20%.

### **Timeline**

- **Launch Date**: July
- **Duration**: July – September

### **Key Actions**

- **Targeted Advertising**: Utilize platforms like Google Ads and Facebook to reach specific demographics.
- **Community Engagement**: Host webinars and live Q&A sessions.
- **Content Creation**: Produce engaging videos and infographics.

### **Supporting Data**

- **Previous Campaign Success**: Our Q2 campaign increased sales by 12%. We will refine similar strategies for Q3.
- **Social Media Growth**: Last year, our Instagram followers grew by 25% during a similar campaign.

### **Conclusion**

We believe this strategy will effectively increase our market share. To achieve these goals, we need your support and collaboration. Let’s work together to make this campaign a success. Please review the proposal and provide your feedback by the end of the week.

Mem0 enables a seamless, intelligent content-writing workflow, perfect for content creators, marketers, and technical writers looking to scale their personal tone and structure across work.

## Help & Resources

- [Mem0 Platform](https://app.mem0.ai/)

<Snippet file="get-help.mdx" />
"""
logger.info("## **Expected Output**")

logger.info("\n\n[DONE]", bright=True)