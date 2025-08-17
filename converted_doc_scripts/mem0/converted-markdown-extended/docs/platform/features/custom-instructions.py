import asyncio
from jet.transformers.formatters import format_json
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
title: Custom Instructions
description: 'Control how Mem0 extracts and stores memories using natural language guidelines'
icon: "pencil"
iconType: "solid"
---

## What are Custom Instructions?

Custom instructions are natural language guidelines that tell Mem0 exactly what information to extract and remember from conversations. Think of them as smart filters that ensure your AI application captures only the most relevant data for your specific use case.

<CodeGroup>
"""
logger.info("## What are Custom Instructions?")

prompt = """
Extract only health and wellness information:
- Symptoms, medications, and treatments
- Exercise routines and dietary habits
- Doctor appointments and health goals

Exclude: Personal identifiers, financial data
"""

client.project.update(custom_instructions=prompt)

"""

"""

prompt = `
Extract only health and wellness information:
- Symptoms, medications, and treatments
- Exercise routines and dietary habits
- Doctor appointments and health goals

Exclude: Personal identifiers, financial data
`

async def run_async_code_9925c25d():
    await client.project.update({ custom_instructions: prompt })
    return 
 = asyncio.run(run_async_code_9925c25d())
logger.success(format_json())

"""
</CodeGroup>

## Why Use Custom Instructions?

- **Focus on What Matters**: Only capture information relevant to your application
- **Maintain Privacy**: Explicitly exclude sensitive data like passwords or personal identifiers
- **Ensure Consistency**: All memories follow the same extraction rules across your project
- **Improve Quality**: Filter out noise and irrelevant conversations

## How to Set Custom Instructions

### Basic Setup

<CodeGroup>
"""
logger.info("## Why Use Custom Instructions?")

client.project.update(custom_instructions="Your guidelines here...")

response = client.project.get(fields=["custom_instructions"])
logger.debug(response["custom_instructions"])

"""

"""

async def run_async_code_ae2f12f1():
    await client.project.update({ custom_instructions: "Your guidelines here..." })
    return 
 = asyncio.run(run_async_code_ae2f12f1())
logger.success(format_json())

async def run_async_code_03dab194():
    async def run_async_code_63ebd058():
        response = await client.project.get({ fields: ["custom_instructions"] })
        return response
    response = asyncio.run(run_async_code_63ebd058())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_03dab194())
logger.success(format_json(response))
console.log(response.custom_instructions)

"""
</CodeGroup>

### Best Practice Template

Structure your instructions using this proven template:

Your Task: [Brief description of what to extract]

Information to Extract:
1. [Category 1]:
   - [Specific details]
   - [What to look for]

2. [Category 2]:
   - [Specific details]
   - [What to look for]

Guidelines:
- [Processing rules]
- [Quality requirements]

Exclude:
- [Sensitive data to avoid]
- [Irrelevant information]

## Real-World Examples

<Tabs>
  <Tab title="E-commerce Customer Support">
<CodeGroup>
"""
logger.info("### Best Practice Template")

instructions = """
Extract customer service information for better support:

1. Product Issues:
   - Product names, SKUs, defects
   - Return/exchange requests
   - Quality complaints

2. Customer Preferences:
   - Preferred brands, sizes, colors
   - Shopping frequency and habits
   - Price sensitivity

3. Service Experience:
   - Satisfaction with support
   - Resolution time expectations
   - Communication preferences

Exclude: Payment card numbers, passwords, personal identifiers.
"""

client.project.update(custom_instructions=instructions)

"""

"""

instructions = `
Extract customer service information for better support:

1. Product Issues:
   - Product names, SKUs, defects
   - Return/exchange requests
   - Quality complaints

2. Customer Preferences:
   - Preferred brands, sizes, colors
   - Shopping frequency and habits
   - Price sensitivity

3. Service Experience:
   - Satisfaction with support
   - Resolution time expectations
   - Communication preferences

Exclude: Payment card numbers, passwords, personal identifiers.
`

async def run_async_code_5592c6e5():
    await client.project.update({ custom_instructions: instructions })
    return 
 = asyncio.run(run_async_code_5592c6e5())
logger.success(format_json())

"""
</CodeGroup>
  </Tab>
  <Tab title="Personalized Learning Platform">
<CodeGroup>
"""

education_prompt = """
Extract learning-related information for personalized education:

1. Learning Progress:
   - Course completions and current modules
   - Skills acquired and improvement areas
   - Learning goals and objectives

2. Student Preferences:
   - Learning styles (visual, audio, hands-on)
   - Time availability and scheduling
   - Subject interests and career goals

3. Performance Data:
   - Assignment feedback and patterns
   - Areas of struggle or strength
   - Study habits and engagement

Exclude: Specific grades, personal identifiers, financial information.
"""

client.project.update(custom_instructions=education_prompt)

"""

"""

educationPrompt = `
Extract learning-related information for personalized education:

1. Learning Progress:
   - Course completions and current modules
   - Skills acquired and improvement areas
   - Learning goals and objectives

2. Student Preferences:
   - Learning styles (visual, audio, hands-on)
   - Time availability and scheduling
   - Subject interests and career goals

3. Performance Data:
   - Assignment feedback and patterns
   - Areas of struggle or strength
   - Study habits and engagement

Exclude: Specific grades, personal identifiers, financial information.
`

async def run_async_code_22d7ee77():
    await client.project.update({ custom_instructions: educationPrompt })
    return 
 = asyncio.run(run_async_code_22d7ee77())
logger.success(format_json())

"""
</CodeGroup>
  </Tab>
  <Tab title="AI Financial Advisor">
<CodeGroup>
"""

finance_prompt = """
Extract financial planning information for advisory services:

1. Financial Goals:
   - Retirement and investment objectives
   - Risk tolerance and preferences
   - Short-term and long-term goals

2. Life Events:
   - Career and income changes
   - Family changes (marriage, children)
   - Major planned purchases

3. Investment Interests:
   - Asset allocation preferences
   - ESG or ethical investment interests
   - Previous investment experience

Exclude: Account numbers, SSNs, passwords, specific financial amounts.
"""

client.project.update(custom_instructions=finance_prompt)

"""

"""

financePrompt = `
Extract financial planning information for advisory services:

1. Financial Goals:
   - Retirement and investment objectives
   - Risk tolerance and preferences
   - Short-term and long-term goals

2. Life Events:
   - Career and income changes
   - Family changes (marriage, children)
   - Major planned purchases

3. Investment Interests:
   - Asset allocation preferences
   - ESG or ethical investment interests
   - Previous investment experience

Exclude: Account numbers, SSNs, passwords, specific financial amounts.
`

async def run_async_code_31eb4f8e():
    await client.project.update({ custom_instructions: financePrompt })
    return 
 = asyncio.run(run_async_code_31eb4f8e())
logger.success(format_json())

"""
</CodeGroup>
  </Tab>
</Tabs>

## Advanced Techniques

### Conditional Processing

Handle different conversation types with conditional logic:

<CodeGroup>
"""
logger.info("## Advanced Techniques")

advanced_prompt = """
Extract information based on conversation context:

IF customer support conversation:
- Issue type, severity, resolution status
- Customer satisfaction indicators

IF sales conversation:
- Product interests, budget range
- Decision timeline and influencers

IF onboarding conversation:
- User experience level
- Feature interests and priorities

Always exclude personal identifiers and maintain professional context.
"""

client.project.update(custom_instructions=advanced_prompt)

"""
</CodeGroup>

### Testing Your Instructions

Always test your custom instructions with real messages examples:

<CodeGroup>
"""
logger.info("### Testing Your Instructions")

messages = [
    {"role": "user", "content": "I'm having billing issues with my subscription"},
    {"role": "assistant", "content": "I can help with that. What's the specific problem?"},
    {"role": "user", "content": "I'm being charged twice each month"}
]

result = client.add(messages, user_id="test_user")
memories = client.get_all(user_id="test_user")

for memory in memories:
    logger.debug(f"Extracted: {memory['memory']}")

"""
</CodeGroup>

## Best Practices

### ✅ Do
- **Be specific** about what information to extract
- **Use clear categories** to organize your instructions
- **Test with real conversations** before deploying
- **Explicitly state exclusions** for privacy and compliance
- **Start simple** and iterate based on results

### ❌ Don't
- Make instructions too long or complex
- Create conflicting rules within your guidelines
- Be overly restrictive (balance specificity with flexibility)
- Forget to exclude sensitive information
- Skip testing with diverse conversation examples

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **Instructions too long** | Break into focused categories, keep concise |
| **Missing important data** | Add specific examples of what to capture |
| **Capturing irrelevant info** | Strengthen exclusion rules and be more specific |
| **Inconsistent results** | Clarify guidelines and test with more examples |
"""
logger.info("## Best Practices")

logger.info("\n\n[DONE]", bright=True)