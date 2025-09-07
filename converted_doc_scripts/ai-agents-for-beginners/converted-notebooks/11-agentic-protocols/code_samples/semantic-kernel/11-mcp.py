async def main():
    from jet.transformers.formatters import format_json
    from IPython.display import display, HTML
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
    from semantic_kernel.connectors.mcp import MCPStdioPlugin
    from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent
    from typing import Annotated
    import asyncio
    import json
    import os
    import shutil
    import subprocess
    import sys
    import traceback
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Semantic Kernel with OpenBnB MCP Server Integration
    
    This notebook demonstrates how to use Semantic Kernel with the actual OpenBnB MCP server to search for real Airbnb accommodations using MCPStdioPlugin. For LLM Access, its uses Azure AI Foundry. To setup your environment variables, you can follow the [Setup Lesson ](/00-course-setup/README.md)
    
    ## Import the Needed Packages
    """
    logger.info("# Semantic Kernel with OpenBnB MCP Server Integration")
    
    
    
    
    
    """
    ## Creating the MCP Plugin Connection
    
    We'll connect to the [OpenBnB MCP server](https://github.com/openbnb-org/mcp-server-airbnb) using MCPStdioPlugin. This server provides Airbnb search functionality through the @openbnb/mcp-server-airbnb package.
    
    ## Creating the Client
    
    In this sample, we will use Azure AI Foundry for our LLM access. Make sure your environment variables are set up correctly.
    
    ## Environment Configuration
    
    Configure Azure Ollama settings. Make sure you have the following environment variables set:
    - `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`
    - `AZURE_OPENAI_ENDPOINT`
    # - `AZURE_OPENAI_API_KEY`
    """
    logger.info("## Creating the MCP Plugin Connection")
    
    load_dotenv()
    
    logger.debug("\n🤖 Creating Ollama service...")
    chat_completion_service = OllamaChatCompletion(ai_model_id="llama3.2")
    
    """
    ## Understanding the OpenBnB MCP Integration
    
    This notebook connects to the **real OpenBnB MCP server** that provides actual Airbnb search functionality.
    
    ### How it works:
    
    1. **MCPStdioPlugin**: Uses standard input/output communication with the MCP server
    2. **Real NPM Package**: Downloads and runs `@openbnb/mcp-server-airbnb` via npx
    3. **Live Data**: Returns actual Airbnb property data from their APIs
    4. **Function Discovery**: The agent automatically discovers available functions from the MCP server
    
    ### Available Functions:
    
    The OpenBnB MCP server typically provides:
    - **search_listings** - Search for Airbnb properties by location and criteria
    - **get_listing_details** - Get detailed information about specific properties
    - **check_availability** - Check availability for specific dates
    - **get_reviews** - Retrieve reviews for properties
    - **get_host_info** - Get information about property hosts
    
    ### Prerequisites:
    
    - **Node.js** installed on your system
    - **Internet connection** to download the MCP server package
    - **NPX** available (comes with Node.js)
    
    ### Testing the Connection:
    
    You can test the MCP server manually by running:
    ```bash
    npx -y @openbnb/mcp-server-airbnb
    ```
    
    This will download and start the OpenBnB MCP server, which Semantic Kernel then connects to for real Airbnb data.
    
    ## Running the Agent with OpenBnB MCP Server
    
    Now we will run the AI Agent that connects to the OpenBnB MCP server to search for real Airbnb accommodations in Stockholm for 2 adults and 1 kid. Feel free to change the `user_inputs` list to modify the search criteria.
    """
    logger.info("## Understanding the OpenBnB MCP Integration")
    
    user_inputs = [
        "Find Airbnb in Stockholm for 2 adults 1 kid",
    ]
    
    
    async def main():
        """Main function to run the MCP-enabled agent with real OpenBnB server using Azure Ollama"""
    
        try:
            logger.debug("🚀 Starting with Azure Ollama...")
    
            logger.debug("🔍 Checking Azure environment variables...")
            # required_vars = ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
            # for var in required_vars:
            #     if os.getenv(var):
            #         logger.debug(f"✅ {var} is set")
            #     else:
            #         logger.debug(f"❌ {var} is NOT set")
    
            logger.debug("\n🔧 Creating MCP Plugin...")
    
            async with MCPStdioPlugin(
                    name="AirbnbSearch",
                    description="Search for Airbnb accommodations using OpenBnB MCP server",
                    command="npx",
                    args=["-y", "@openbnb/mcp-server-airbnb"],
            ) as airbnb_plugin:
    
                logger.debug("✅ MCP Plugin created and connected")
    
                await asyncio.sleep(2)
    
                try:
                    tools = await airbnb_plugin.get_tools()
                    logger.success(format_json(tools))
                    logger.debug(f"🔧 Available tools: {[tool.name for tool in tools]}")
                except Exception as e:
                    logger.debug(f"⚠️ Could not list tools: {str(e)}")
    
                agent = ChatCompletionAgent(
                    service=chat_completion_service,
                    name="AirbnbAgent",
                    instructions="""You are an Airbnb search assistant. Use the available functions to search for properties.
                    Format results in a clear HTML table with columns for property name, price, rating, and link.""",
                    plugins=[airbnb_plugin],
                )
    
                logger.debug("✅ Agent created with Azure Ollama")
    
                thread: ChatHistoryAgentThread | None = None
    
                for user_input in user_inputs:
                    logger.debug(f"\n🔍 User: {user_input}")
    
                    try:
                        response = await agent.get_response(messages=user_input, thread=thread)
                        logger.success(format_json(response))
                        thread = response.thread
    
                        response_text = str(response)
    
                        response_text = response_text.replace('```html', '').replace('```', '')
    
                        logger.debug(f"🤖 {response.name}: {response_text[:200]}..." if len(response_text) > 200 else response_text)
    
                        if '<table' in response_text.lower():
                            table_css = """
                            <style>
                                .airbnb-results table {
                                    border-collapse: collapse;
                                    width: 100%;
                                    margin: 10px 0;
                                }
                                .airbnb-results th, .airbnb-results td {
                                    border: 1px solid #ddd;
                                    padding: 8px;
                                    text-align: left;
                                }
                                .airbnb-results th {
                                    background-color: #f2f2f2;
                                    font-weight: bold;
                                }
                                .airbnb-results tr:nth-child(even) {
                                    background-color: #f9f9f9;
                                }
                                .airbnb-results a {
                                    color: #1976d2;
                                    text-decoration: none;
                                }
                                .airbnb-results a:hover {
                                    text-decoration: underline;
                                }
                            </style>
                            """
                            html_output = f'{table_css}<div class="airbnb-results">{response_text}</div>'
                            display(HTML(html_output))
                        else:
                            display(HTML(f'<div class="airbnb-results">{response_text}</div>'))
    
                    except Exception as e:
                        logger.debug(f"❌ Error processing user input: {str(e)}")
                        traceback.print_exc()
    
                if thread:
                    await thread.delete()
                    logger.debug("🧹 Thread cleaned up")
    
        except Exception as e:
            logger.debug(f"❌ Main error: {str(e)}")
            traceback.print_exc()
    
    logger.debug("🚀 Starting MCP Agent...")
    await main()
    logger.debug("✅ Done!")
    
    """
    # Summary
    Congratulations! You've successfully built an AI agent that integrates with real-world accommodation search using the Model Context Protocol (MCP):
    
    ## Technologies Used:
    - Semantic Kernel - For building intelligent agents with Azure Ollama
    - Azure AI Foundry - For LLM capabilities and chat completion
    - MCP (Model Context Protocol) - For standardized tool integration
    - OpenBnB MCP Server - For real Airbnb search functionality
    - Node.js/NPX - For running the external MCP server
    
    ## What You've Learned:
    - MCP Integration: Connecting Semantic Kernel agents to external MCP servers
    - Real-time Data Access: Searching actual Airbnb properties through live APIs
    - Protocol Communication: Using stdio communication between agent and MCP server
    - Function Discovery: Automatically discovering available functions from MCP servers
    - Streaming Responses: Capturing and logging function calls in real-time
    - HTML Rendering: Formatting agent responses with styled tables and interactive displays
    
    ## Next Steps:
    - Integrate additional MCP servers (weather, flights, restaurants)
    - Build a multi-agent system combining MCP and A2A protocols
    - Create custom MCP servers for your own data sources
    - Implement persistent conversation memory across sessions
    - Deploy the agent to Azure Functions with MCP server orchestration
    - Add user authentication and booking capabilities
    
    
    """
    logger.info("# Summary")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())