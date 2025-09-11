from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from toolbox_langchain import ToolboxClient
import os
import shutil

async def main():
    
    
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
    # MCP Toolbox for Databases
    
    Integrate your databases with LangChain agents using MCP Toolbox.
    
    ## Overview
    
    [MCP Toolbox for Databases](https://github.com/googleapis/genai-toolbox) is an open source MCP server for databases. It was designed with enterprise-grade and production-quality in mind. It enables you to develop tools easier, faster, and more securely by handling the complexities such as connection pooling, authentication, and more.
    
    Toolbox Tools can be seemlessly integrated with Langchain applications. For more
    information on [getting
    started](https://googleapis.github.io/genai-toolbox/getting-started/local_quickstart/) or
    [configuring](https://googleapis.github.io/genai-toolbox/getting-started/configure/)
    MCP Toolbox, see the
    [documentation](https://googleapis.github.io/genai-toolbox/getting-started/introduction/).
    
    ![architecture](https://raw.githubusercontent.com/googleapis/genai-toolbox/refs/heads/main/docs/en/getting-started/introduction/architecture.png)
    
    ## Setup
    
    This guide assumes you have already done the following:
    
    1. Installed [Python 3.9+](https://wiki.python.org/moin/BeginnersGuide/Download) and [pip](https://pip.pypa.io/en/stable/installation/).
    2. Installed [PostgreSQL 16+ and the `psql` command-line client](https://www.postgresql.org/download/).
    
    ### 1. Setup your Database
    
    First, let's set up a PostgreSQL database. We'll create a new database, a dedicated user for MCP Toolbox, and a `hotels` table with some sample data.
    
    Connect to PostgreSQL using the `psql` command. You may need to adjust the command based on your PostgreSQL setup (e.g., if you need to specify a host or a different superuser).
    
    ```bash
    psql -U postgres
    ```
    
    Now, run the following SQL commands to create the user, database, and grant the necessary permissions:
    
    ```sql
    CREATE USER toolbox_user WITH PASSWORD 'my-password';
    CREATE DATABASE toolbox_db;
    GRANT ALL PRIVILEGES ON DATABASE toolbox_db TO toolbox_user;
    ALTER DATABASE toolbox_db OWNER TO toolbox_user;
    ```
    
    Connect to your newly created database with the new user:
    
    ```sql
    \c toolbox_db toolbox_user
    ```
    
    Finally, create the `hotels` table and insert some data:
    
    ```sql
    CREATE TABLE hotels(
      id            INTEGER NOT NULL PRIMARY KEY,
      name          VARCHAR NOT NULL,
      location      VARCHAR NOT NULL,
      price_tier    VARCHAR NOT NULL,
      booked        BIT     NOT NULL
    );
    
    INSERT INTO hotels(id, name, location, price_tier, booked)
    VALUES 
      (1, 'Hilton Basel', 'Basel', 'Luxury', B'0'),
      (2, 'Marriott Zurich', 'Zurich', 'Upscale', B'0'),
      (3, 'Hyatt Regency Basel', 'Basel', 'Upper Upscale', B'0');
    ```
    You can now exit `psql` by typing `\q`.
    
    ### 2. Install MCP Toolbox
    
    Next, we will install MCP Toolbox, define our tools in a `tools.yaml` configuration file, and run the MCP Toolbox server.
    
    For **macOS** users, the easiest way to install is with [Homebrew](https://formulae.brew.sh/formula/mcp-toolbox):
    
    ```bash
    brew install mcp-toolbox
    ```
    
    For other platforms, [download the latest MCP Toolbox binary for your operating system and architecture.](https://github.com/googleapis/genai-toolbox/releases)
    
    Create a `tools.yaml` file. This file defines the data sources MCP Toolbox can connect to and the tools it can expose to your agent. For production use, always use environment variables for secrets.
    
    ```yaml
    sources:
      my-pg-source:
        kind: postgres
        host: 127.0.0.1
        port: 5432
        database: toolbox_db
        user: toolbox_user
        password: my-password
    
    tools:
      search-hotels-by-location:
        kind: postgres-sql
        source: my-pg-source
        description: Search for hotels based on location.
        parameters:
          - name: location
            type: string
            description: The location of the hotel.
        statement: SELECT id, name, location, price_tier FROM hotels WHERE location ILIKE '%' || $1 || '%';
      book-hotel:
        kind: postgres-sql
        source: my-pg-source
        description: >-
            Book a hotel by its ID. If the hotel is successfully booked, returns a confirmation message.
        parameters:
          - name: hotel_id
            type: integer
            description: The ID of the hotel to book.
        statement: UPDATE hotels SET booked = B'1' WHERE id = $1;
    
    toolsets:
      hotel_toolset:
        - search-hotels-by-location
        - book-hotel
    ```
    
    Now, in a separate terminal window, start the MCP Toolbox server. If you installed via Homebrew, you can just run `toolbox`. If you downloaded the binary manually, you'll need to run `./toolbox` from the directory where you saved it:
    
    ```bash
    toolbox --tools-file "tools.yaml"
    ```
    
    MCP Toolbox will start on `http://127.0.0.1:5000` by default and will hot-reload if you make changes to your `tools.yaml` file.
    
    ## Instantiation
    """
    logger.info("# MCP Toolbox for Databases")
    
    # !pip install toolbox-langchain
    
    
    with ToolboxClient("http://127.0.0.1:5000") as client:
        search_tool = await client.aload_tool("search-hotels-by-location")
        logger.success(format_json(search_tool))
    
    """
    ## Invocation
    """
    logger.info("## Invocation")
    
    
    with ToolboxClient("http://127.0.0.1:5000") as client:
        search_tool = await client.aload_tool("search-hotels-by-location")
        logger.success(format_json(search_tool))
        results = search_tool.invoke({"location": "Basel"})
        logger.debug(results)
    
    """
    ## Use within an agent
    
    Now for the fun part! We'll install the required LangChain packages and create an agent that can use the tools we defined in MCP Toolbox.
    """
    logger.info("## Use within an agent")
    
    # %pip install -U --quiet toolbox-langchain langgraph langchain-google-vertexai
    
    """
    With the packages installed, we can define our agent. We will use `ChatVertexAI` for the model and `ToolboxClient` to load our tools. The `create_react_agent` from `langgraph.prebuilt` creates a robust agent that can reason about which tools to call.
    
    **Note:** Ensure your MCP Toolbox server is running in a separate terminal before executing the code below.
    """
    logger.info("With the packages installed, we can define our agent. We will use `ChatVertexAI` for the model and `ToolboxClient` to load our tools. The `create_react_agent` from `langgraph.prebuilt` creates a robust agent that can reason about which tools to call.")
    
    
    prompt = """
    You're a helpful hotel assistant. You handle hotel searching and booking.
    When the user searches for a hotel, list the full details for each hotel found: id, name, location, and price tier.
    Always use the hotel ID for booking operations.
    For any bookings, provide a clear confirmation message.
    Don't ask for clarification or confirmation from the user; perform the requested action directly.
    """
    
    
    async def run_queries(agent_executor):
        config = {"configurable": {"thread_id": "hotel-thread-1"}}
    
        query1 = "I need to find a hotel in Basel."
        logger.debug(f'\n--- USER: "{query1}" ---')
        inputs1 = {"messages": [("user", prompt + query1)]}
        for event in agent_executor.stream_events(
            inputs1, config=config, version="v2"
        ):
            if event["event"] == "on_chat_model_end" and event["data"]["output"].content:
                logger.debug(f"--- AGENT: ---\n{event['data']['output'].content}")
    
        query2 = "Great, please book the Hyatt Regency Basel for me."
        logger.debug(f'\n--- USER: "{query2}" ---')
        inputs2 = {"messages": [("user", query2)]}
        for event in agent_executor.stream_events(
            inputs2, config=config, version="v2"
        ):
            if event["event"] == "on_chat_model_end" and event["data"]["output"].content:
                logger.debug(f"--- AGENT: ---\n{event['data']['output'].content}")
    
    """
    ## Run the agent
    """
    logger.info("## Run the agent")
    
    async def main():
        await run_hotel_agent()
    
    
    async def run_hotel_agent():
        model = ChatVertexAI(model_name="gemini-2.5-flash")
    
        async with ToolboxClient("http://127.0.0.1:5000") as client:
                tools = await client.aload_toolset("hotel_toolset")
            
                agent = create_react_agent(model, tools, checkpointer=MemorySaver())
            
                await run_queries(agent)
            
            
        logger.success(format_json(result))
    await main()
    
    """
    You've successfully connected a LangChain agent to a local database using MCP Toolbox! 🥳
    
    ## API reference
    
    The primary class for this integration is `ToolboxClient`.
    
    For more information, see the following resources:
    - [Toolbox Official Documentation](https://googleapis.github.io/genai-toolbox/)
    - [Toolbox GitHub Repository](https://github.com/googleapis/genai-toolbox)
    - [Toolbox LangChain SDK](https://github.com/googleapis/mcp-toolbox-python-sdk/tree/main/packages/toolbox-langchain)
    
    MCP Toolbox has a variety of features to make developing Gen AI tools for databases seamless:
    - [Authenticated Parameters](https://googleapis.github.io/genai-toolbox/resources/tools/#authenticated-parameters): Bind tool inputs to values from OIDC tokens automatically, making it easy to run sensitive queries without potentially leaking data
    - [Authorized Invocations](https://googleapis.github.io/genai-toolbox/resources/tools/#authorized-invocations): Restrict access to use a tool based on the users Auth token
    - [OpenTelemetry](https://googleapis.github.io/genai-toolbox/how-to/export_telemetry/): Get metrics and tracing from MCP Toolbox with [OpenTelemetry](https://opentelemetry.io/docs/)
    
    # Community and Support
    
    We encourage you to get involved with the community:
    - ⭐️ Head over to the [GitHub repository](https://github.com/googleapis/genai-toolbox) to get started and follow along with updates.
    - 📚 Dive into the [official documentation](https://googleapis.github.io/genai-toolbox/getting-started/introduction/) for more advanced features and configurations.
    - 💬 Join our [Discord server](https://discord.com/invite/a4XjGqtmnG) to connect with the community and ask questions.
    """
    logger.info("## API reference")
    
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