messages: List[ChatMessage] = [{"role": "user", "content": "Say 'Hello, world!' in one sentence."}]

# When
result = llm.chat(messages, temperature=0.0)