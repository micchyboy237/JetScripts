from typing import Dict, Any, List
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.adapters.swarms.ollama_function_caller2 import OllamaFunctionCaller, Message

def example_single_task() -> str:
    """
    Demonstrates running a single task with OllamaFunctionCaller.
    Returns:
        str: The assistant's response to the math task.
    """
    ollama_assistant = OllamaFunctionCaller(
        model_name="llama3.2",
        agent_name="MathAssistant",
        name="Ollama Math Tutor",
        description="A math tutor powered by Ollama",
        instructions="You are a helpful math tutor. Solve math problems accurately.",
        temperature=0.1,
    )
    task = "What is 3 + 5?"
    response = ollama_assistant.run(task)
    print(f"Single Task: {task}\nResponse: {response}")
    return response

def example_chat_method() -> str:
    """
    Demonstrates using the chat method with a list of messages.
    Returns:
        str: The assistant's response to the chat request.
    """
    ollama_assistant = OllamaFunctionCaller(
        model_name="llama3.2",
        agent_name="MathAssistant",
        name="Ollama Math Tutor",
        description="A math tutor powered by Ollama",
        instructions="You are a helpful math tutor.",
        temperature=0.1,
    )
    messages = [
        Message(role="system", content="You are a math tutor."),
        Message(role="user", content="Explain how to solve 2x + 3 = 7"),
    ]
    response = ollama_assistant.chat(messages)
    print(f"Chat Request: {messages[1].content}\nResponse: {response}")
    return response

def example_generate_method() -> str:
    """
    Demonstrates using the generate method with a prompt.
    Returns:
        str: The assistant's response to the generate request.
    """
    ollama_assistant = OllamaFunctionCaller(
        model_name="llama3.2",
        agent_name="MathAssistant",
        name="Ollama Math Tutor",
        description="A math tutor powered by Ollama",
        temperature=0.1,
    )
    prompt = "Simplify the expression 4x + 2x - 3"
    response = ollama_assistant.generate(prompt)
    print(f"Generate Prompt: {prompt}\nResponse: {response}")
    return response

def example_function_calling() -> str:
    """
    Demonstrates adding and calling a custom function with OllamaFunctionCaller.
    Returns:
        str: The result of the function call.
    """
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b

    ollama_assistant = OllamaFunctionCaller(
        model_name="llama3.2",
        agent_name="MathAssistant",
        name="Ollama Math Tutor",
        description="A math tutor powered by Ollama",
        temperature=0.1,
    )
    ollama_assistant.add_function(
        func=calculate_sum,
        description="Calculates the sum of two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    )
    task = '{"function": "calculate_sum", "arguments": {"a": 10, "b": 20}}'
    response = ollama_assistant.run(task)
    print(f"Function Call Task: {task}\nResponse: {response}")
    return response

def example_batch_run() -> List[str]:
    """
    Demonstrates running multiple tasks sequentially with batch_run.
    Returns:
        List[str]: List of responses from the assistant.
    """
    ollama_assistant = OllamaFunctionCaller(
        model_name="llama3.2",
        agent_name="MathAssistant",
        name="Ollama Math Tutor",
        description="A math tutor powered by Ollama",
        instructions="You are a helpful math tutor. Solve math problems accurately.",
        temperature=0.1,
    )
    tasks = [
        "Solve 2x = 8",
        "What is 7 * 6?",
        "Explain the Pythagorean theorem",
    ]
    responses = ollama_assistant.batch_run(tasks)
    for task, response in zip(tasks, responses):
        print(f"Batch Task: {task}\nResponse: {response}")
    return responses

def example_run_concurrently() -> List[str]:
    """
    Demonstrates running multiple tasks concurrently with run_concurrently.
    Returns:
        List[str]: List of responses from the assistant.
    """
    ollama_assistant = OllamaFunctionCaller(
        model_name="llama3.2",
        agent_name="MathAssistant",
        name="Ollama Math Tutor",
        description="A math tutor powered by Ollama",
        instructions="You are a helpful math tutor. Solve math problems accurately.",
        temperature=0.1,
    )
    tasks = [
        "Solve 2x = 8",
        "What is 7 * 6?",
        "Explain the Pythagorean theorem",
    ]
    responses = ollama_assistant.run_concurrently(tasks)
    for task, response in zip(tasks, responses):
        print(f"Concurrent Task: {task}\nResponse: {response}")
    return responses

def example_list_models() -> List[str]:
    """
    Demonstrates listing available models with OllamaFunctionCaller.
    Returns:
        List[str]: List of available model names.
    """
    ollama_assistant = OllamaFunctionCaller(
        model_name="llama3.2",
        agent_name="MathAssistant",
        name="Ollama Math Tutor",
        description="A math tutor powered by Ollama",
        temperature=0.1,
    )
    models = ollama_assistant.list_models()
    print(f"Available Models: {models}")
    return models

def example_show_model() -> Dict[str, Any]:
    """
    Demonstrates showing details of the current model.
    Returns:
        Dict[str, Any]: Details of the current model.
    """
    ollama_assistant = OllamaFunctionCaller(
        model_name="llama3.2",
        agent_name="MathAssistant",
        name="Ollama Math Tutor",
        description="A math tutor powered by Ollama",
        temperature=0.1,
    )
    model_details = ollama_assistant.show_model()
    print(f"Model Details: {model_details}")
    return model_details

if __name__ == "__main__":
    print("=== Running OllamaFunctionCaller Usage Examples ===")
    example_single_task()
    print("\n")
    example_chat_method()
    print("\n")
    example_generate_method()
    print("\n")
    example_function_calling()
    print("\n")
    example_batch_run()
    print("\n")
    example_run_concurrently()
    print("\n")
    example_list_models()
    print("\n")
    example_show_model()