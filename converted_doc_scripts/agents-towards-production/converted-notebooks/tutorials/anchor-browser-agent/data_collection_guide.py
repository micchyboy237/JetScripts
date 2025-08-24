import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from playwright.async_api import async_playwright
import asyncio
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
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--anchor-browser-agent--anchor_browser_data_collection_guide)

# Anchor Browser Agent Guide: Data Collection from Grafana Dashboards

A comprehensive guide for integrating Anchor Browser Agent into production-ready AI systems, focusing on automated data collection from web-based dashboards and monitoring interfaces.

## Introduction

Modern AI agents often need to interact with web applications that were designed for human users rather than programmatic access. While many services offer APIs, a significant portion of critical business data remains locked behind web interfaces - dashboards, monitoring tools, and administrative panels that require browser-based interaction.

Anchor Browser Agent addresses this challenge by providing a cloud-based browser automation platform specifically designed for AI agents. Unlike traditional web scraping tools that rely on fragile selectors and require extensive maintenance, Anchor leverages built-in AI capabilities to understand and interact with web interfaces contextually.

This guide demonstrates how to build a robust data collection system that can extract structured information from complex web dashboards, using Grafana as our primary example. The techniques covered here apply broadly to any web-based monitoring or administrative interface.

### Why This Approach Matters

Traditional web automation approaches face several critical limitations:
- **Maintenance Overhead**: CSS selectors break when interfaces change
- **Scale Challenges**: Managing browser infrastructure is complex and costly
- **Dynamic Content**: JavaScript-heavy applications often require sophisticated timing and interaction patterns
- **Anti-Bot Detection**: Modern websites employ increasingly sophisticated bot detection mechanisms

Anchor Browser Agent solves these problems by providing enterprise-grade browser automation as a service, with AI-powered navigation that adapts to interface changes and handles complex interaction patterns automatically.

## Key Benefits and Capabilities
 
 Anchor Browser Agent offers:
 
 - **Cloud-based browsers**: No local setup - sessions run in isolated, reliable cloud environments.
 - **AI-powered web interaction**: Use natural language instructions instead of fragile selectors ([docs](https://docs.anchorbrowser.io/agentic-browser-control/ai-task-completion?utm_source=agents-towards-production)).
 - **Advanced session management**: Persist authentication, customize browser fingerprints, and control timeouts ([docs](https://docs.anchorbrowser.io/api-reference/browser-sessions/start-browser-session?utm_source=agents-towards-production)).
 - **Proxy & geo support**: Access geo-restricted content and avoid rate limits with residential/mobile proxies ([docs](https://docs.anchorbrowser.io/advanced/proxy?utm_source=agents-towards-production)).
 - **Session recording**: Record browser sessions for debugging and compliance ([docs](https://docs.anchorbrowser.io/essentials/recording?utm_source=agents-towards-production)).
 - **Profile management**: Persist browser identities for complex authentication flows ([docs](https://docs.anchorbrowser.io/essentials/authentication-and-identity?utm_source=agents-towards-production)).

## Architecture Overview

Understanding how Anchor Browser Agent works will help you design more effective integrations with your AI systems.

<img src="./assets/data-collection-diagram.png" alt="Anchor Browser Agent Architecture" width="600"/>

The architecture follows a cloud-native design where your AI agent communicates with remote browser instances through a combination of REST APIs and real-time protocols:

1. **Session Creation**: Your agent uses Anchor's REST API to create configured browser sessions
2. **Browser Control**: Direct connection to browser instances via Chrome DevTools Protocol (CDP)
3. **AI Navigation**: Built-in AI agents handle complex web interactions based on natural language instructions
4. **Data Extraction**: Structured data is returned to your agent for further processing

This separation of concerns allows for both simple automation tasks and sophisticated interaction patterns while maintaining the reliability and scalability required for production systems.

## System Requirements
Python 3.11 or higher is required for optimal performance and compatibility. Newer Python versions provide:

## Getting Started: Account Setup and Authentication

Setting up your Anchor Browser Agent integration requires obtaining API credentials and configuring your development environment.

### Creating Your Anchor Browser Account

Navigate to [anchorbrowser.io](https://anchorbrowser.io?utm_source=agents-towards-production) to create your account. The platform offers a free tier that provides sufficient quota for development and testing of automation workflows.

<img src="./assets/signup.png" alt="Account Creation Process" width="300"/>

### Generating API Credentials

After account creation, visit the [API Access Page](https://app.anchorbrowser.io/api-access?utm_source=agents-towards-production) to generate your API key. This key will authenticate all requests to the Anchor platform and should be treated as a sensitive credential.

<img src="./assets/dashboard.png" alt="API Key Generation Interface" width="500"/>

### Security Considerations

When handling API keys in production environments:
- Store keys in environment variables or secure configuration management systems
- Never commit API keys to version control systems
- Implement key rotation policies for long-running production systems
- Monitor API usage to detect unauthorized access

### Installing Required Dependencies

The Anchor Browser Agent integration requires two primary Python packages that work together to provide comprehensive browser automation capabilities.

## Dependency Installation and Configuration

Install the required Python packages for Anchor Browser Agent integration:
"""
logger.info("# Anchor Browser Agent Guide: Data Collection from Grafana Dashboards")

# !pip3 install playwright requests

"""
### Understanding the Dependency Stack

**Playwright** serves as the bridge between your Python code and browser instances:
- Provides a robust Chrome DevTools Protocol (CDP) implementation
- Handles real-time communication with remote browser sessions
- Offers comprehensive error handling and connection management
- Supports both synchronous and asynchronous operation modes

**Requests** manages HTTP communication with Anchor's REST API:
- Session creation and configuration
- Authentication and credential management
- Session lifecycle operations (start, monitor, terminate)

This two-layer approach separates session management (REST API) from real-time browser control (CDP), providing both simplicity for basic operations and power for complex automation scenarios.

## Use Case: Automated Data Collection from Grafana Dashboards

This section demonstrates a practical implementation of Anchor Browser Agent for extracting structured data from web-based monitoring dashboards. We'll use Grafana as our example, but the principles apply to any dynamic web interface.

### Understanding the Challenge

Grafana dashboards present several automation challenges that make them ideal for demonstrating Anchor's capabilities:
- **Dynamic Content Loading**: Charts and metrics load asynchronously via JavaScript
- **Interactive Elements**: Some data only appears through hover actions or click interactions
- **Authentication Requirements**: Production dashboards often require SSO or complex login flows
- **Real-Time Updates**: Data refreshes continuously, requiring careful timing considerations

### Business Value

Automated dashboard monitoring enables AI agents to:
- Continuously monitor infrastructure health without human intervention
- Correlate metrics across multiple monitoring systems
- Detect anomalies and patterns that might be missed in manual reviews
- Generate automated reports and alerts based on visual dashboard data

### Target Dashboard Overview

Our example focuses on a Kubernetes monitoring dashboard that displays critical infrastructure metrics:

<img src="./assets/grafana-dashboard.png" alt="Example Grafana Dashboard" width="800"/>

This dashboard contains:
- Node performance metrics and resource utilization
- CPU and memory usage percentages across cluster nodes
- Network and storage performance indicators
- Alert status and health indicators

The goal is to extract this visual information as structured data that can be processed by downstream AI systems.

## Implementation: Browser Session Creation

The first step in any Anchor Browser automation is creating a properly configured browser session. Session configuration significantly impacts performance, cost, and reliability.
"""
logger.info("### Understanding the Dependency Stack")


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
logger.debug("Session created successfully:")
logger.debug(json.dumps(session_data, indent=2))

connection_string = session_data['data']['cdp_url']
logger.debug(f"\nBrowser connection URL: {connection_string}")

"""
### Understanding Session Configuration

**Timeout Settings**: The timeout configuration balances cost control with operational flexibility. The `max_duration` prevents runaway sessions that could incur unexpected charges, while `idle_timeout` ensures efficient resource cleanup when automation completes.

**Viewport Configuration**: Setting a large viewport (1920x1080) ensures that dashboard elements are fully visible and properly rendered. Many responsive web applications adjust their layout based on viewport size, so consistent sizing prevents layout-related automation failures.

**Recording Features**: Enabling session recording provides valuable debugging capabilities. Video recordings allow you to see exactly what the browser encountered during automation, making it easier to diagnose failures and optimize interaction patterns.

**Connection URL**: The `cdp_url` returned in the response provides direct access to the browser instance via Chrome DevTools Protocol. This URL enables real-time control and monitoring of the browser session.

## Implementation: AI-Powered Data Extraction

With the browser session established, we can now implement the core data extraction functionality using Anchor's built-in AI capabilities.
"""
logger.info("### Understanding Session Configuration")


async def async_func_3():
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
        
        dashboard_url = "https://play.grafana.org/a/grafana-k8s-app/navigation/nodes?from=now-1h&to=now&refresh=1m"
        
        logger.debug(f"Navigating to dashboard: {dashboard_url}")
        async def run_async_code_2973204e():
            await page.goto(dashboard_url, wait_until="domcontentloaded")
        asyncio.run(run_async_code_2973204e())
        
        logger.debug("Waiting for dashboard content to load...")
        async def run_async_code_baeaee97():
            await asyncio.sleep(5)
        asyncio.run(run_async_code_baeaee97())
        
        extraction_prompt = 'Collect the node names and their CPU average %, return in JSON array'
        
        logger.debug("AI agent analyzing dashboard content...")
        async def run_async_code_157f398d():
            result = await ai_agent.evaluate(extraction_prompt)
            return result
        result = asyncio.run(run_async_code_157f398d())
        logger.success(format_json(result))
        
        logger.debug("\nExtracted data:")
        logger.debug(result)
asyncio.run(async_func_3())

"""
### Understanding the AI Extraction Process

**Browser Connection**: The `connect_over_cdp` method establishes a direct connection to the cloud browser instance. This provides real-time control capabilities similar to what you would have with a local browser, but with the benefits of cloud infrastructure.

**AI Agent Integration**: Anchor's built-in AI agent runs as a service worker within the browser context. This agent has been trained to understand web interfaces and can execute natural language instructions to interact with complex layouts.

**Navigation Strategy**: Using `wait_until="domcontentloaded"` ensures the basic page structure is ready before proceeding. The additional sleep period allows time for JavaScript-driven content (like Grafana charts) to render completely.

**Natural Language Instructions**: Instead of writing complex selectors or interaction scripts, you can provide natural language instructions to the AI agent. The agent interprets these instructions contextually and adapts to the specific layout it encounters.

**Async/Await Pattern**: The code uses Python's async/await syntax to handle the asynchronous nature of browser automation. This is particularly important in Jupyter notebooks, which have specific requirements for async code execution.

### Synchronous Alternative for Standard Python Scripts

The previous example uses `async_playwright` for Jupyter notebook compatibility. In standard Python scripts, you can use the synchronous version for simpler code structure:
"""
logger.info("### Understanding the AI Extraction Process")

logger.debug("Note: Use the synchronous version above in standard Python scripts.")
async def run_async_code_617445ea():
    logger.debug("Remove 'await' keywords and use 'sync_playwright' instead of 'async_playwright'.")
asyncio.run(run_async_code_617445ea())

"""
## Production Best Practices

When deploying Anchor Browser Agent in production environments, following these best practices will ensure reliable, maintainable, and cost-effective automation.


### Session Lifecycle Management
Use [session timeout configurations](https://docs.anchorbrowser.io/advanced/session-termination?utm_source=agents-towards-production) appropriate for your use case. Implement proper session cleanup to avoid unnecessary charges. Consider session pooling for high-frequency operations.

### Authentication and State Management
For dashboards requiring authentication, leverage [persistent browser profiles](https://docs.anchorbrowser.io/essentials/authentication-and-identity?utm_source=agents-towards-production) to maintain login state across multiple automation runs. This eliminates the need to re-authenticate for each session.

### Batch Processing Optimization
When extracting data from multiple dashboards or performing related tasks, batch these operations within single sessions to optimize resource usage and reduce overhead.

### Security and Credential Management
Store API keys and other sensitive credentials in environment variables or secure configuration management systems. Implement key rotation policies and monitor API usage for security anomalies.
"""
logger.info("## Production Best Practices")

logger.info("\n\n[DONE]", bright=True)