ollama_request = {
    "model": "llama3.2",
    "stream": False,
    "prompt": None,
    "messages": [
        {
            "role": "user",
            "content": "What is three plus one?"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "add_two_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "required": [
                        "a",
                        "b"
                    ],
                    "properties": {
                        "a": {
                            "type": "integer",
                            "description": "The first number"
                        },
                        "b": {
                            "type": "integer",
                            "description": "The second number"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "subtract_two_numbers",
                "description": "Subtract two numbers",
                "parameters": {
                    "type": "object",
                    "required": [
                        "a",
                        "b"
                    ],
                    "properties": {
                        "a": {
                            "type": "integer",
                            "description": "The first number"
                        },
                        "b": {
                            "type": "integer",
                            "description": "The second number"
                        }
                    }
                }
            }
        }
    ]
}
