import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from playwright.async_api import async_playwright
import json
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--anchor-browser-agent--anchor_browser_data_entry_guide)

# Anchor Browser Agent Guide: Automated Form Submission and Data Entry

A comprehensive guide for implementing intelligent form automation using Anchor Browser Agent, enabling AI systems to complete complex web forms with human-like accuracy and contextual understanding.

## Introduction
 
 Automating form submission is a common but complex task, especially for multi-step forms with dynamic fields and conditional logic. Traditional tools often fall short when forms require contextual understanding and adaptability.
 
 Anchor Browser Agent solves these challenges by combining cloud-based browser automation with AI that understands both the source data and the form structure. This allows AI agents to extract information from documents and complete web forms intelligently, as shown in our charity donation example.
 
 ### Business Impact
 
 Automated form submission with AI enables:
 - High-volume, error-free data entry
 - Integration with systems lacking APIs
 - Scalable workflows without extra human resources
 - Consistent data formatting and validation
 
 These benefits are valuable for onboarding, application processing, data migration, and any workflow requiring web-based data transfer.

## Why Use Anchor for Form Automation?

 - **No Local Setup:** Run form automation in the cloud - no browser installs or infrastructure headaches.
 - **AI-Driven:** Built-in AI agents understand form fields and context, so you don’t need to manually map every field.
 - **Handles Real-World Forms:** Supports authentication, multi-step forms, and session management out of the box.
 - **Works Anywhere:** Use proxies for region-specific forms and compliance.
 - **Audit-Ready:** Automatic session recording for debugging and compliance.

## Architecture Overview: Form Automation Workflow

<img src="./assets/data-entry-diagram.png" alt="Form Automation Architecture" width="600"/>

## System Requirements and Development Environment
Python 3.11 or higher provides the optimal foundation for Anchor Browser Agent integration:

## Getting Started: Account Setup and Authentication

Establishing your Anchor Browser Agent integration requires proper account configuration and credential management.

### Account Creation and Service Tiers

Navigate to [anchorbrowser.io](https://anchorbrowser.io?utm_source=agents-towards-production) to create your account. The platform offers multiple service tiers designed to accommodate different automation volumes and complexity requirements.

<img src="./assets/signup.png" alt="Account Registration Process" width="300"/>

The free tier provides sufficient capabilities for development, testing, and proof-of-concept form automation workflows. Production deployments may require higher tiers depending on volume and advanced feature requirements.

### API Key Generation and Management

Access the [API Access Page](https://app.anchorbrowser.io/api-access?utm_source=agents-towards-production) to generate your authentication credentials. These keys provide access to all Anchor platform capabilities and should be managed according to enterprise security standards.

<img src="./assets/dashboard.png" alt="API Credential Management" width="500"/>

### Production Security Practices

When implementing form automation in production environments:
- Use environment-based credential management systems
- Implement credential rotation policies for long-running systems
- Monitor API usage patterns to detect anomalous activity
- Segregate credentials by environment (development, staging, production)
- Implement least-privilege access principles for different automation workflows

### Dependency Installation and Configuration

Form automation requires specific Python packages that provide browser control and AI integration capabilities.

## Dependency Installation for Form Automation

Install the required packages for comprehensive form automation capabilities:
"""
logger.info("# Anchor Browser Agent Guide: Automated Form Submission and Data Entry")

# !pip3 install playwright requests

"""
### Understanding the Form Automation Stack

**Playwright** serves as the foundation for sophisticated form interactions:
- Provides precise form field targeting and data entry capabilities
- Handles complex form validation and error recovery scenarios
- Supports dynamic form elements and JavaScript-driven interactions
- Offers comprehensive event handling for form submission workflows

**Requests** manages the broader automation workflow:
- Session creation and configuration for form automation contexts
- Authentication and credential management for protected forms
- Session lifecycle management to optimize resource utilization

This combination provides both high-level workflow management and low-level form interaction precision, enabling robust automation of even the most complex form scenarios.

## Use Case: Intelligent Resume-to-Form Data Transfer

This implementation demonstrates a sophisticated form automation scenario where an AI agent must extract information from an unstructured document and intelligently populate a web form. This use case showcases Anchor's ability to handle complex data interpretation and contextual form completion.

### Understanding the Automation Challenge

Our example involves an AI agent that must:
1. **Document Analysis**: Read and understand a personal resume with unstructured formatting
2. **Information Extraction**: Identify relevant personal and contact information
3. **Context Understanding**: Determine which extracted data is appropriate for form fields
4. **Form Navigation**: Navigate to and interact with a charity donation form
5. **Intelligent Completion**: Populate form fields based on contextual understanding
6. **Validation Handling**: Respond to form validation and complete the submission process

### Business Application Scenarios

This type of automation enables:
- **Customer Onboarding**: Automatically populate application forms from submitted documents
- **Data Migration**: Transfer information between systems with different data formats
- **Compliance Processing**: Complete regulatory forms using data from internal systems
- **Lead Processing**: Convert prospect information into structured CRM entries

### Target Form Overview

Our demonstration uses a charity donation form that requires personal information, contact details, and donation preferences:

<video autoPlay muted loop playsInline src="./assets/vid.mp4" alt="Charity Form Automation Process" width="800"/>

This form represents typical challenges in web form automation:
- Multiple field types (text, email, dropdown, checkbox)
- Client-side validation requirements
- Conditional fields based on user selections
- Required field validation and error handling

## Implementation: Session Creation for Form Automation

Creating an optimally configured browser session is crucial for reliable form automation. Session parameters directly impact form interaction success rates and automation reliability.
"""
logger.info("### Understanding the Form Automation Stack")


url = "https://api.anchorbrowser.io/v1/sessions"

payload = {
    "session": {
        "timeout": {
            "max_duration": 4,  # Allow sufficient time for multi-page forms
            "idle_timeout": 2   # Quick cleanup after form completion
        },
        "recording": {
            "active": True,    # Record session for debugging purposes, Default is True
        }
    },
    "browser": {
        "viewport": {
            "width": 1920,   # Ensure form elements are fully visible
            "height": 1080   # Accommodate complex form layouts
        }
    }
}

headers = {
    "anchor-api-key": "<ANCHOR-API-KEY>",  # Replace with your actual API key
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

session_data = response.json()
logger.debug("Form automation session created:")
logger.debug(json.dumps(session_data, indent=2))

connection_string = session_data['data']['cdp_url']
session_id = session_data['data']['id']

logger.debug(f"\nSession ID: {session_id}")
logger.debug(f"Connection URL: {connection_string[:50]}...")

if 'live_view_url' in session_data['data']:
    logger.debug(f"Live View: {session_data['data']['live_view_url']}")

"""
### Understanding Session Configuration for Forms

**Timeout Strategy for Form Workflows**: Form automation requires different timeout considerations than data collection. The `max_duration` allows sufficient time for multi-page forms and user interface delays, while `idle_timeout` ensures efficient cleanup after form completion.

**Viewport Optimization for Forms**: Form elements can be sensitive to viewport size, with some forms adjusting their layout or hiding elements on smaller screens. The 1920x1080 configuration ensures consistent form rendering across different websites.

**Recording for Form Compliance**: Form automation often involves sensitive data and compliance requirements. Session recordings provide audit trails showing exactly how forms were completed, which can be crucial for regulatory compliance and error diagnosis.

**Live View Monitoring**: The live view URL allows real-time observation of form completion processes, which is particularly valuable during development and when debugging complex form interactions.

## Implementation: AI-Powered Form Completion

This implementation demonstrates Anchor's advanced AI task completion capabilities, where the system can understand both source documents and target forms to perform intelligent data transfer.
"""
logger.info("### Understanding Session Configuration for Forms")


async def async_func_4():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(connection_string)
        
        context = browser.contexts[0]
        
        ai_agent_url = "chrome-extension://bppehibnhionalpjigdjdilknbljaeai/background.js"
        ai_agent = next((sw for sw in context.service_workers if sw.url == ai_agent_url), None)
        
        if not ai_agent:
            raise Exception("AI agent not found in browser context")
        
        async def run_async_code_fdeb03ab():
            page = context.pages[0] if context.pages else await context.new_page()
        asyncio.run(run_async_code_fdeb03ab())
        
        logger.debug("Navigating to source document for data extraction...")
        resume_url = 'https://www.wix.com/demone2/nicol-rider'
        async def run_async_code_685c043c():
            await page.goto(resume_url)
        asyncio.run(run_async_code_685c043c())
        
        logger.debug("Document loaded, initiating AI-powered form completion...")
        
        task_configuration = {
            "prompt": (
                "Read the resume on this page, understand the personal details, "
                "and complete the form at https://formspree.io/library/donation/charity-donation-form/preview.html "
                "as if you were this person. Use appropriate information from the resume "
                "to fill out personal details. Limit the donation amount to $10."
            ),
        
            "provider": "openai",
            "model": "gpt-4",
        
            "highlight_elements": True
        }
        
        logger.debug("AI agent analyzing document and preparing form completion...")
        
        async def run_async_code_5b9fc313():
            result = await ai_agent.evaluate(json.dumps(task_configuration))
            return result
        result = asyncio.run(run_async_code_5b9fc313())
        logger.success(format_json(result))
        
        logger.debug("\nForm automation completed successfully!")
        logger.debug(f"AI Agent Result: {result}")
asyncio.run(async_func_4())

logger.info("\n\n[DONE]", bright=True)