import asyncio
from auto_memory import Filter


def mock_event_emitter(event: dict):
    """
    Mock event emitter to simulate real-world usage of event handling.
    """
    print(f"Event Emitted: {event}")


async def main():
    """
    Main function to demonstrate and test Filter functionalities.
    """
    # Initialize the Filter class
    filter_instance = Filter()

    # Sample user and message data for testing
    mock_user = {
        "id": "12345",
        "valves": {"show_status": True},
    }

    mock_body = {
        "messages": [
            {"role": "system", "content": "Welcome to the memory service."},
            {
                "role": "user",
                "content": "Please remember that my favorite fruit is mango.",
            },
            {"role": "assistant", "content": "Understood."},
        ]
    }

    # Test `inlet` functionality
    inlet_result = filter_instance.inlet(
        mock_body, mock_event_emitter, mock_user)
    print("Inlet Result:")
    print(inlet_result)

    # Test `outlet` functionality
    outlet_result = await filter_instance.outlet(
        mock_body, mock_event_emitter, mock_user
    )
    print("Outlet Result:")
    print(outlet_result)

    # Test `identify_memories` functionality
    sample_text = "I love hiking and exploring nature trails during weekends."
    identified_memories = await filter_instance.identify_memories(sample_text)
    print("Identified Memories:")
    print(identified_memories)

    # Test querying the OpenAI-compatible API (mock setup required for testing)
    model = "llama3.1:latest"
    system_prompt = "Identify useful memories."
    prompt = "User enjoys painting landscapes in their free time."
    try:
        api_response = await filter_instance.query_openai_api(model, system_prompt, prompt)
        print("API Response:")
        print(api_response)
    except Exception as e:
        print(f"API Query Failed: {e}")

    # Test `process_memories` (dependent on implementation of `store_memory`)
    mock_memories = '["User likes cycling", "User is learning Spanish"]'
    process_result = await filter_instance.process_memories(mock_memories, mock_user)
    print("Processed Memories Result:")
    print(process_result)


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
