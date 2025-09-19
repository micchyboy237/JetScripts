async def main():
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import (
        FunctionAgent,
        AgentWorkflow,
        ToolCallResult,
        AgentStream,
    )
    from llama_index.core.memory import Memory
    from llama_index.tools.artifact_editor import (
        ArtifactEditorToolSpec,
        ArtifactMemoryBlock,
    )
    from pydantic import BaseModel, Field
    import json
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    # Build an Order Completion Agent with an Artifact Editor Tool
    
    In this example, we'll build a chat assistant that is designed to fill in a custom 'form'.
    
    As an example use-case, we'll build an order taking assistant that will need to get a few set pieces of information from the end-user before proceeding. Like their delivery address, and the contents of their order.
    
    To build this, we're using the new `ArtifactEditorToolSpec` and `ArtifactMemoryBlock`
    """
    logger.info("# Build an Order Completion Agent with an Artifact Editor Tool")

    # !pip install llama-index llama-index-tools-artifact-editor

    # from getpass import getpass

    # if "OPENAI_API_KEY" not in os.environ:
    #     os.environ["OPENAI_API_KEY"] = getpass("OllamaFunctionCalling API Key: ")

    class Pizza(BaseModel):
        name: str = Field(description="The name of the pizza")
        remove: list[str] | None = Field(
            description="If exists, the ingredients the customer requests to remove",
            default=None,
        )
        add: list[str] | None = Field(
            description="If exists, the ingredients the customer requests to be added",
            default=None,
        )

    class Address(BaseModel):
        street_address: str = Field(
            description="The street address of the customer"
        )
        city: str = Field(description="The city of the customer")
        state: str = Field(description="The state of the customer")
        zip_code: str = Field(description="The zip code of the customer")

    class Order(BaseModel):
        pizzas: list[Pizza] | None = Field(
            description="The pizzas ordered by the customer", default=None
        )
        address: Address | None = Field(
            description="The full address of the customer", default=None
        )

    tool_spec = ArtifactEditorToolSpec(Order)
    tools = tool_spec.to_tool_list()

    memory = Memory.from_defaults(
        session_id="order_editor",
        memory_blocks=[ArtifactMemoryBlock(artifact_spec=tool_spec)],
        token_limit=60000,
        chat_history_token_ratio=0.7,
    )

    llm = OllamaFunctionCalling(model="llama3.2")

    agent = AgentWorkflow(
        agents=[
            FunctionAgent(
                llm=llm,
                tools=tools,
                system_prompt="""You are a worker at a Pizzeria. Your job is to talk to users and gather order information.
                At every step, you should check the order completeness before responding to the user, and ask for any possibly
                missing information.""",
            )
        ],
    )

    async def chat():
        while True:
            user_msg = input("User: ").strip()
            if user_msg.lower() in ["exit", "quit"]:
                logger.debug("\n------ORDER COMPLETION-------\n")
                logger.debug(
                    f"The Order was placed with the following Order schema:\n: {json.dumps(tool_spec.get_current_artifact(), indent=4)}"
                )
                break

            handler = agent.run(user_msg, memory=memory)
            async for ev in handler.stream_events():
                if isinstance(ev, AgentStream):
                    logger.debug(ev.delta, end="", flush=True)
                elif isinstance(ev, ToolCallResult):
                    logger.debug(
                        f"\n\nCalling tool: {ev.tool_name} with kwargs: {ev.tool_kwargs}"
                    )

            logger.debug("\n\nCurrent artifact: ",
                         tool_spec.get_current_artifact())

    await chat()

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
