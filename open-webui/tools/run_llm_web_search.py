import asyncio


async def example_emit():
    async def mock_event_emitter(data):
        print(f"Event Emitted: {data}")

    await emit_status(mock_event_emitter, "Processing request", False)
    await emit_message(mock_event_emitter, "The operation is complete.")

# Run the example
asyncio.run(example_emit())
