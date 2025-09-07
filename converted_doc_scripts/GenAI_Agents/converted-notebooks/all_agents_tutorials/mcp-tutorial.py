async def main():
    from jet.transformers.formatters import format_json
    from anthropic import Ollama
    from jet.logger import CustomLogger
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from typing import List, Dict, Any
    import json
    import os
    import re
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Building an Agent with MCP: Seamless Integration of AI and External Resources
    
    ## Introduction
    
    Model Context Protocol (MCP) is an open protocol designed to standardize how applications provide context to large language models (LLMs). Think of MCP like a USB-C port for AI applications - just as USB-C provides a standardized way to connect devices to various peripherals, MCP provides a standardized way to connect AI models to different data sources and tools.
    
    This tutorial will guide you through implementing MCP in your AI agent applications, demonstrating how it can enhance your agents' capabilities by providing seamless access to external resources, tools, and data sources.
    
    ## Why MCP Matters for Agents
    
    Traditional methods of connecting AI models with external resources often involve custom integrations for each data source or tool. This leads to:
    
    - **Integration Complexity**: Each new data source requires a unique implementation
    - **Scalability Issues**: Adding new tools becomes progressively harder
    - **Maintenance Overhead**: Updates to one integration may break others
    
    MCP solves these challenges by providing a standardized protocol that enables:
    
    - **Unified Access**: A single interface for multiple data sources and tools
    - **Plug-and-Play Extensions**: Easy addition of new capabilities
    - **Stateful Communication**: Real-time, two-way communication between AI and resources
    - **Dynamic Discovery**: AI can find and use new tools on the fly
    
    Here's a concise paragraph highlighting the official MCP Server examples:
    
    ## Official MCP Server Examples
    
    The MCP community maintains a collection of reference server implementations that showcase best practices and demonstrate various integration patterns. These official examples, available at [MCP Servers](https://github.com/modelcontextprotocol/servers/tree/main/src), provide valuable starting points for developers looking to create their own MCP servers.
    
    ## What We'll Build
    
    In this tutorial, we'll implement:
    
    1. **Build Your MCP Servers and Use It**: Build a MCP server with customized tools and connect to Claude Desktop
    2. **Customized Tool-Enabled Agent**: Create an customized agent that can use external tools via MCP
    
    By the end of this tutorial, you'll understand how MCP can enhance your AI agents by providing them with access to the broader digital ecosystem, making them more capable, context-aware, and useful.
    
    Let's begin by understanding the MCP architecture and setting up our environment!
    
    ## MCP Architecture Overview
    
    ![MCP Architecture](../images/mcp_architecture.png)
    
    MCP follows a client-server architecture with three main components:
    
    - **Host**: The AI application (like Claude Desktop, Cursor or a customized agent) that needs access to external resources
    - **Clients**: Connectors that maintain connections with servers
    - **Servers**: Lightweight programs that expose capabilities (data, tools, prompts) via the MCP protocol
    - **Data Sources**: Both local (files, databases) and remote services (APIs) that MCP servers can access
    
    Communication within MCP uses JSON-RPC 2.0 over WebSocket connections, ensuring real-time, bidirectional communication between components.
    
    ## Experiencing MCP: Try Before You Build
    
    While this tutorial focuses on building your own MCP servers and integrating them with AI agents, you might want to quickly experience how MCP works in practice before diving into development.
    
    The official MCP documentation provides an excellent quick start guide for users who want to try existing MCP servers with Claude Desktop or other compatible AI applications. This gives you a hands-on feel for the capabilities MCP enables without writing any code.
    
    **👉 Try it yourself:** [MCP Quick Start Guide for Users](https://modelcontextprotocol.io/quickstart/user)
    
    By exploring the quick start guide, you'll gain practical insight into what we're building in this tutorial. When you're ready to understand the inner workings and create your own implementations, continue with our step-by-step development process below.
    
    Now, let's start building our own MCP server and client!
    
    ## Building Your MCP Server
    
    Now that we understand the basics of MCP, let's build our first MCP server! In this section, we'll create a cryptocurrency price lookup service using the CoinGecko API. Our server will provide tools that allow an AI to check the current price or market data of cryptocurrencies.
    
    ### Setting Up Our Environment
    
    Before we dive into implementation, let's install the necessary packages and set up our environment.
    
    > **Note:** For the installation steps, please open a terminal window. These commands should be run in a regular terminal, not in a Jupyter notebook cell.
    
    #### Step 1: Install uv Package Manager
    
    ```bash
    # Run this in your terminal, not in Jupyter
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    
    #### Step 2: Set up the Project
    
    ```bash
    # Create and navigate to a project directory
    mkdir mcp-crypto-server
    cd mcp-crypto-server
    uv init
    
    # Create and activate virtual environment
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    
    # Install dependencies
    uv add "mcp[cli]" httpx
    ```
    
    ### Running the MCP Server
    
    After we set up the envirnment, we can start build our tools.
    
    Please checkout [mcp_server.py](scripts/mcp_server.py) to see how to build tools.
    
    
    now we can start the server by runnning following commands in the ternimal:
    ```bash
    # Copy the server file from the scripts folder
    cp ../scripts/mcp_server.py .
    
    # Start the MCP server 
    uv run mcp_server.py
    ```
    
    ### Integration with Claude Desktop
    If you haven't download Claude Desktop, checkout [this page](https://claude.ai/download).
    
    To connect your MCP server to Claude Desktop:
    
    #### Step 1: Find the absolute path to your uv command:
    ```bash
    which uv
    ```
    Copy the output (e.g., /user/local/bin/uv or similar)
    
    #### Step 2: Create or edit the Claude Desktop configuration file:
    
    - On macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
    - On Windows: %APPDATA%\Claude\claude_desktop_config.json
    - On Linux: ~/.config/Claude/claude_desktop_config.json
    
    You can checkout [this page](https://modelcontextprotocol.io/quickstart/user#2-add-the-filesystem-mcp-server) to see how to create a config file.
    
    #### Step 3: Add your MCP server configuration:
    
    ```json
    {
        "mcpServers": {
            "crypto-price-tracker": {
                "command": "/ABSOLUTE/PATH/TO/uv",
                "args": [
                    "--directory",
                    "/ABSOLUTE/PATH/TO/GenAI_Agents/all_agents_tutorials/mcp-crypto-server",
                    "run",
                    "mcp_server.py"
                ]
            }
        }
    }
    ```
    Replace `/ABSOLUTE/PATH/TO/uv` with the path you got from the `which uv` command, and `/ABSOLUTE/PATH/TO/GenAI_Agents` with the absolute path to your repository.
    
    
    #### Step 4: Restart Claude Desktop for the changes to take effect.
    
    You should see this hammer in your chat box.
    
    ![Claude Desktop connected with MCP](../images/Claude_Desktop_with_MCP.png)
    
    #### Step 5: Try ask the price of Bitcoin
    
    Type in "What is the current price of Bitcoin ?", and you will get response like:
    
    ![Track Bitcoin price with MCP](../images/track_bitcoin_price_with_mcp.png)
    
    
    Congrats! You've successfully apply your MCP server and tool. Now, you can try add your own tools to [mcp_server.py](/mcp-crypto-server/mcp_server.py). Here is an example:
    
    ```python
    @mcp.tool()
    async def get_crypto_market_info(crypto_ids: str, currency: str = "usd") -> str:
        """
    logger.info("# Building an Agent with MCP: Seamless Integration of AI and External Resources")
        Get market information for one or more cryptocurrencies.
        
        Parameters:
        - crypto_ids: Comma-separated list of cryptocurrency IDs (e.g., 'bitcoin,ethereum')
        - currency: The currency to display values in (default: 'usd')
        
        Returns:
        - Market information including price, market cap, volume, and price changes
        """
        # Construct the API URL
        url = f"{COINGECKO_BASE_URL}/coins/markets"
        
        # Set up the query parameters
        params = {
            "vs_currency": currency,  # Currency to display values in
            "ids": crypto_ids,        # Comma-separated crypto IDs
            "order": "market_cap_desc", # Order by market cap
            "page": 1,                # Page number
            "sparkline": "false"      # Exclude sparkline data
        }
        
        try:
            # Make the API call
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                # Parse the response
                data = response.json()
                
                # Check if we got any data
                if not data:
                    return f"No data found for cryptocurrencies: '{crypto_ids}'. Please check the IDs and try again."
                
                # Format the results
                result = ""
                for crypto in data:
                    name = crypto.get('name', 'Unknown')
                    symbol = crypto.get('symbol', '???').upper()
                    price = crypto.get('current_price', 'Unknown')
                    market_cap = crypto.get('market_cap', 'Unknown')
                    volume = crypto.get('total_volume', 'Unknown')
                    price_change = crypto.get('price_change_percentage_24h', 'Unknown')
                    
                    result += f"{name} ({symbol}):\n"
                    result += f"Current price: {price} {currency.upper()}\n"
                    result += f"Market cap: {market_cap} {currency.upper()}\n"
                    result += f"24h trading volume: {volume} {currency.upper()}\n"
                    result += f"24h price change: {price_change}%\n\n"
                
                return result
                
        except Exception as e:
            return f"Error fetching market data: {str(e)}"
    ```
    
    Rerun your mcp server with `uv run mcp_server.py`, restart Claude Desktop, and type "What's the market data for Dogecoin and Solana?". You will get the response like this:
    
    ![Track Crypto Market Data with MCP](../images/track_crypto_market_data_with_mcp.png)
    
    ## Customized Agent executing tool via MCP
    
    After we build our own MCP, let's try building MCP Host & Client ourselves.
    
    ### Understanding the Architecture
    
    In this section, we'll build our own MCP Host and Client. Unlike the previous approach where we connected to Claude Desktop, we'll now create our own agent that can:
    1. Act as an MCP Host
    2. Discover available tools from our MCP server
    3. Understand when to use which tool based on user queries
    4. Execute tools with appropriate parameters
    5. Process tool results to provide helpful responses
    
    This architecture follows a pattern common in modern AI systems:
    - **Discovery Phase**: Our custom host discovers what tools are available
    - **Planning Phase**: The agent decides which tool to use based on the user's query
    - **Execution Phase**: Our client connects to the server and executes the selected tool
    - **Interpretation Phase**: The agent explains the results in natural language
    
    Here is a simple worflow diagram:
    
    ![Track Crypto Market Data with MCP](../images/customized_mcp_host.png)
    
    Important Reminder Before Running the Code:
    ⚠️ Don't forget to start your MCP server first! ⚠️
    Before running the agent code in following tutorial, make sure your MCP server is up and running. Otherwise, your agent won't have any tools to discover or execute.
    
    Let's start by setting up our environment and importing the necessary libraries:
    """
    logger.info("# Construct the API URL")
    
    # ! pip install mcp anthropic
    
    """
    We need two primary libraries:
    - **MCP**: To handle the client-server communication with our MCP server, allowing us to build both the host and client components
    - **Ollama**: To interact with Claude, which will power our agent's reasoning capabilities
    
    Now, let's set up the necessary imports and configurations for our agent:
    """
    logger.info("We need two primary libraries:")
    
    
    
    
    # os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key_here"
    
    client = Ollama()
    
    mcp_server_path = "absolute/path/to/your/running/mcp/server"
    logger.debug("Setup complete!")
    
    """
    We're using the `stdio_client` interface from MCP, which allows us to connect to MCP servers that run as separate processes and communicate via standard input/output. This is a simple and robust approach for local development. By implementing both sides of the MCP protocol (host and client), we gain complete control over how our agent interacts with MCP tools.
    
    ### Tool Discovery: Building Our MCP Host
    
    The first step in building our custom MCP implementation is to create a host that can discover what tools are available from our MCP server. Our host will act as the intermediary between the user, the AI, and the available tools - similar to how Claude Desktop functions, but under our complete control.
    
    Let's implement a function to connect to our MCP server and discover its tools:
    """
    logger.info("### Tool Discovery: Building Our MCP Host")
    
    async def discover_tools():
        """
        Connect to the MCP server and discover available tools.
        Returns information about the available tools.
        """
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        SEP = "=" * 40
    
        server_params = StdioServerParameters(
            command="python",  # Command to run the server
            args=[mcp_server_path],  # Path to your MCP server script
        )
    
        logger.debug(f"{BLUE}{SEP}\n🔍 DISCOVERY PHASE: Connecting to MCP server...{RESET}")
    
        async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    logger.debug(f"{BLUE}📡 Initializing MCP connection...{RESET}")
                    await session.initialize()
            
                    logger.debug(f"{BLUE}🔎 Discovering available tools...{RESET}")
                    tools = await session.list_tools()
            
                    tool_info = []
                    for tool_type, tool_list in tools:
                        if tool_type == "tools":
                            for tool in tool_list:
                                tool_info.append({
                                    "name": tool.name,
                                    "description": tool.description,
                                    "schema": tool.inputSchema
                                })
            
                    logger.debug(f"{GREEN}✅ Successfully discovered {len(tool_info)} tools{RESET}")
                    logger.debug(f"{SEP}")
                    return tool_info
            
        logger.success(format_json(result))
    logger.debug("Tool discovery function defined")
    
    """
    This function acts as our host's discovery component:
    
    1. **Creates Server Parameters**: Configures how to launch and connect to the MCP server
    2. **Establishes Connection**: Uses `stdio_client` to create a communication channel
    3. **Initializes Session**: Sets up the MCP session using the communication channel
    4. **Discovers Tools**: Calls `list_tools()` to get all available tools
    5. **Formats Results**: Converts the tools into a more usable format for our agent
    
    We're using an asynchronous approach (`async/await`) because MCP operations are non-blocking by design. This is important in a host implementation, as it allows our agent to handle multiple operations concurrently and remain responsive even when waiting for tool operations to complete.
    
    Let's test our tool discovery function to make sure it works properly:
    """
    logger.info("This function acts as our host's discovery component:")
    
    tools = await discover_tools()
    logger.success(format_json(tools))
    logger.debug(f"Discovered {len(tools)} tools:")
    for i, tool in enumerate(tools, 1):
        logger.debug(f"{i}. {tool['name']}: {tool['description']}")
    
    """
    When we run this code, we should see a list of the tools available from our MCP server. In this case, we're expecting to see our cryptocurrency tools.
    
    ### Tool Execution: Implementing Our MCP Client
    
    Now that our host can discover available tools, we need to implement the client component that can execute them. Unlike third-party tools that might have this functionality built-in, we're creating our own client to execute MCP tools with complete control and transparency:
    """
    logger.info("### Tool Execution: Implementing Our MCP Client")
    
    async def execute_tool(tool_name: str, arguments: Dict[str, Any]):
        """
        Execute a specific tool provided by the MCP server.
    
        Args:
            tool_name: The name of the tool to execute
            arguments: A dictionary of arguments to pass to the tool
    
        Returns:
            The result from executing the tool
        """
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        SEP = "-" * 40
    
        server_params = StdioServerParameters(
            command="python",
            args=[mcp_server_path],
        )
    
        logger.debug(f"{YELLOW}{SEP}")
        logger.debug(f"⚙️ EXECUTION PHASE: Running tool '{tool_name}'")
        logger.debug(f"📋 Arguments: {json.dumps(arguments, indent=2)}")
        logger.debug(f"{SEP}{RESET}")
    
        async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
            
                    logger.debug(f"{BLUE}📡 Sending request to MCP server...{RESET}")
                    result = await session.call_tool(tool_name, arguments)
            
                    logger.debug(f"{GREEN}✅ Tool execution complete{RESET}")
            
                    result_preview = str(result)
                    if len(result_preview) > 150:
                        result_preview = result_preview[:147] + "..."
            
                    logger.debug(f"{BLUE}📊 Result: {result_preview}{RESET}")
                    logger.debug(f"{SEP}")
            
                    return result
            
        logger.success(format_json(result))
    logger.debug("Tool execution function defined")
    
    """
    This function forms the core of our MCP client:
    
    1. **Connects to Server**: Similar to our discovery function, it establishes a connection to the MCP server
    2. **Executes Tool**: Calls the specified tool with the provided arguments
    3. **Returns Result**: Gives back whatever the tool returns
    
    Notice that for each tool execution, we create a new connection to the MCP server. While this may seem inefficient, it ensures clean separation between tool calls and avoids potential state issues. This stateless approach simplifies our implementation and makes it more robust. In a production system, you might optimize this by maintaining a persistent connection, but the current approach is excellent for educational purposes as it clearly separates each step in the process.
    
    Now that we have functions to discover and execute tools, we need to integrate these with an AI that can determine when and how to use them. This is where Claude comes in.
    
    ### Integrating AI with Our MCP Implementation
    
    With our host and client components in place, we now need to integrate them with an AI system that can make intelligent decisions about tool usage. This is the "brains" of our custom MCP host, and it needs to:
    1. Understand when a tool is needed based on user input
    2. Choose the appropriate tool for the task
    3. Format the arguments correctly
    4. Process and explain the results
    
    Let's implement a function that orchestrates this entire process:
    """
    logger.info("### Integrating AI with Our MCP Implementation")
    
    async def query_claude(prompt: str, tool_info: List[Dict], previous_messages=None):
        """
        Send a query to Claude and process the response.
    
        Args:
            prompt: User's query
            tool_info: Information about available tools
            previous_messages: Previous messages for maintaining context
    
        Returns:
            Claude's response, potentially after executing tools
        """
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        PURPLE = "\033[95m"
        RESET = "\033[0m"
        SEP = "=" * 40
    
        if previous_messages is None:
            previous_messages = []
    
        logger.debug(f"{PURPLE}{SEP}")
        logger.debug("🧠 REASONING PHASE: Processing query with Claude")
        logger.debug(f"🔤 Query: \"{prompt}\"")
        logger.debug(f"{SEP}{RESET}")
    
        tool_descriptions = "\n\n".join([
            f"Tool: {tool['name']}\nDescription: {tool['description']}\nSchema: {json.dumps(tool['schema'], indent=2)}"
            for tool in tool_info
        ])
    
        system_prompt = f"""You are an AI assistant with access to specialized tools through MCP (Model Context Protocol).
    
    Available tools:
    {tool_descriptions}
    
    When you need to use a tool, respond with a JSON object in the following format:
    {{
        "tool": "tool_name",
        "arguments": {{
            "arg1": "value1",
            "arg2": "value2"
        }}
    }}
    
    Do not include any other text when using a tool, just the JSON object.
    For regular responses, simply respond normally.
    """
    
        filtered_messages = [msg for msg in previous_messages if msg["role"] != "system"]
    
        messages = filtered_messages.copy()
    
        messages.append({"role": "user", "content": prompt})
    
        logger.debug(f"{BLUE}📡 Sending request to Claude API...{RESET}")
    
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            system=system_prompt,  # System prompt as a separate parameter
            messages=messages      # Only user and assistant messages
        )
    
        claude_response = response.content[0].text
        logger.debug(f"{GREEN}✅ Received response from Claude{RESET}")
    
        try:
            json_match = re.search(r'(\{[\s\S]*\})', claude_response)
    
            if json_match:
                json_str = json_match.group(1)
                logger.debug(f"{YELLOW}🔍 Tool usage detected in response{RESET}")
                logger.debug(f"{BLUE}📦 Extracted JSON: {json_str}{RESET}")
    
                tool_request = json.loads(json_str)
    
                if "tool" in tool_request and "arguments" in tool_request:
                    tool_name = tool_request["tool"]
                    arguments = tool_request["arguments"]
    
                    logger.debug(f"{YELLOW}🔧 Claude wants to use tool: {tool_name}{RESET}")
    
                    tool_result = await execute_tool(tool_name, arguments)
                    logger.success(format_json(tool_result))
    
                    if not isinstance(tool_result, str):
                        tool_result = str(tool_result)
    
                    messages.append({"role": "assistant", "content": claude_response})
                    messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
    
                    logger.debug(f"{PURPLE}🔄 Getting Claude's interpretation of the tool result...{RESET}")
    
                    final_response = client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=4000,
                        system=system_prompt,
                        messages=messages
                    )
    
                    logger.debug(f"{GREEN}✅ Final response ready{RESET}")
                    logger.debug(f"{SEP}")
    
                    return final_response.content[0].text, messages
    
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug(f"{YELLOW}⚠️ No tool usage detected in response: {str(e)}{RESET}")
    
        logger.debug(f"{GREEN}✅ Response ready{RESET}")
        logger.debug(f"{SEP}")
    
        return claude_response, messages
    
    logger.debug("Claude query function defined")
    
    """
    This function completes our custom MCP host implementation with a sophisticated reasoning and execution flow:
    
    1. **Tool Description**: We format the tool information in a way Claude can understand
    2. **System Prompt**: We provide instructions on when and how to use tools
    3. **Response Analysis**: We look for JSON tool requests in Claude's responses
    4. **Tool Execution**: If a tool request is detected, we use our client to execute the appropriate tool
    5. **Result Processing**: We send the tool results back to Claude for interpretation
    6. **Conversation Management**: We maintain context by tracking messages
    
    This creates a powerful synergy: Claude provides the reasoning and communication skills, while our MCP tools provide specialized capabilities and real-time data access.
    
    Let's test our agent with a simple query about Bitcoin prices:
    """
    logger.info("This function completes our custom MCP host implementation with a sophisticated reasoning and execution flow:")
    
    query = "What is the current price of Bitcoin?"
    logger.debug(f"Sending query: {query}")
    
    response, messages = await query_claude(query, tools)
    logger.success(format_json(response, messages))
    logger.debug(f"\nAssistant's response:\n{response}")
    
    """
    When we run this query, our complete MCP implementation follows this flow:
    1. Claude (via our host) recognizes this as a request about Bitcoin prices
    2. Our AI decides to use the `get_crypto_price` tool
    3. It formats the arguments correctly (using "bitcoin" as the crypto_id)
    4. Our client connects to the server and executes the tool, returning the current Bitcoin price
    5. Claude explains the result in natural language with additional context
    
    This demonstrates the full capability of our agent: understanding the user's intent, selecting the appropriate tool, executing it correctly, and providing a helpful, context-rich response.
    
    ### Direct Tool Execution via Our Client
    
    While our integrated MCP host typically decides which tools to use based on the user's query, sometimes we might want to directly use our client to execute a specific tool. This is useful for testing our client implementation or demonstrating specific tool functionality. Let's create a simple example:
    """
    logger.info("### Direct Tool Execution via Our Client")
    
    try:
        if tools:
            first_tool = tools[0]
            tool_name = first_tool["name"]
    
            arguments = {"crypto_id": "bitcoin"}
    
            logger.debug(f"Executing tool '{tool_name}' with arguments: {arguments}")
            result = await execute_tool(tool_name, arguments)
            logger.success(format_json(result))
            logger.debug(f"Tool result: {result}")
        else:
            logger.debug("No tools discovered to test")
    except Exception as e:
        logger.debug(f"Error executing tool: {str(e)}")
    
    """
    This direct execution approach is useful for:
    - Testing our client implementation in isolation
    - Debugging tool functionality
    - Building specialized workflows where tool execution is predetermined
    - Verifying that our MCP client works correctly before integrating it with the AI
    
    Now, let's create an interactive chat interface that uses our complete MCP host implementation:
    
    ### Building an Interactive MCP Host Interface
    
    For a complete MCP host implementation, we need a user interface that maintains context across multiple turns of conversation. This allows our host to remember previous interactions and build on them in subsequent exchanges, just like professional MCP hosts such as Claude Desktop. Let's implement a simple chat session function:
    """
    logger.info("### Building an Interactive MCP Host Interface")
    
    async def chat_session():
        """
        Run an interactive chat session with the AI agent.
        """
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        SEP = "=" * 50
    
        logger.debug(f"{CYAN}{BOLD}{SEP}")
        logger.debug("🤖 INITIALIZING MCP AGENT")
        logger.debug(f"{SEP}{RESET}")
    
        try:
            if 'tools' not in globals() or not tools:
                logger.debug(f"{BLUE}🔍 No tools found, discovering available tools...{RESET}")
                tools_local = await discover_tools()
                logger.success(format_json(tools_local))
            else:
                tools_local = tools
    
            logger.debug(f"{GREEN}✅ Agent ready with {len(tools_local)} tools:{RESET}")
    
            for i, tool in enumerate(tools_local, 1):
                logger.debug(f"{YELLOW}  {i}. {tool['name']}{RESET}")
                logger.debug(f"     {tool['description'].strip()}")
    
            logger.debug(f"\n{CYAN}{BOLD}{SEP}")
            logger.debug(f"💬 INTERACTIVE CHAT SESSION")
            logger.debug(f"{SEP}")
            logger.debug(f"Type 'exit' or 'quit' to end the session{RESET}")
    
            messages = []
    
            while True:
                user_input = input(f"\n{BOLD}You:{RESET} ")
    
                if user_input.lower() in ['exit', 'quit']:
                    logger.debug(f"\n{GREEN}Ending chat session. Goodbye!{RESET}")
                    break
    
                logger.debug(f"\n{BLUE}Processing...{RESET}")
                response, messages = await query_claude(user_input, tools_local, messages)
                logger.success(format_json(response, messages))
    
                logger.debug(f"\n{BOLD}Assistant:{RESET} {response}")
    
        except Exception as e:
            logger.debug(f"\n{YELLOW}⚠️ An error occurred: {str(e)}{RESET}")
    
    logger.debug("Chat session function defined. Run 'await chat_session()' in the next cell to start chatting.")
    
    """
    Our MCP host interface:
    
    1. **Initializes Tools**: Our host discovers available tools when starting
    2. **Creates a Session Loop**: Continuously prompts for user input
    3. **Maintains Context**: Passes previous messages to each query, maintaining stateful conversations
    4. **Handles Graceful Exit**: Allows the user to end the session gracefully
    
    This creates a natural, conversational experience where the agent can remember previous interactions. For example, if a user asks about Bitcoin and then follows up with "How about Ethereum?", the agent understands the context.
    
    Now, let's run our chat session to see the complete agent in action:
    
    You may try what we ask in Clude Desktop: What's the market data for Dogecoin and Solana?
    """
    logger.info("Our MCP host interface:")
    
    await chat_session()
    
    """
    With this final piece, we've created a complete custom MCP host implementation that can:
    1. Connect to an MCP server as a client
    2. Discover available tools
    3. Intelligently select and use those tools to answer user queries
    4. Maintain context across a conversation
    
    This demonstrates the power of implementing our own MCP host and client - we get complete control over how AI interacts with tools while maintaining all the benefits of the MCP protocol's standardization.
    
    ## Conclusion:
    
    The Model Context Protocol represents a transformative approach to integrating AI models with external resources, solving critical challenges in AI application development:
    
    ### Protocol Advantages
    - **Standardized Integration**: Eliminates complex, custom API connections
    - **Dynamic Tool Discovery**: Enables AI to find and use tools seamlessly
    - **Flexible Communication**: Supports real-time, bidirectional interactions
    
    ### Technical Highlights
    Our implementation demonstrated:
    - Building MCP server with specialized tools
    - Creating a host that can dynamically discover and execute tools
    - Integrating AI model with external resources
    
    ### Key Citations
    
    - [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
    - [MCP Quick Start Guide](https://modelcontextprotocol.io/quickstart/user)
    - [CoinGecko API Documentation](https://www.coingecko.com/en/api)
    """
    logger.info("## Conclusion:")
    
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