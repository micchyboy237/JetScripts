from litellm import completion
from jet.logger import logger

# params = {
#     'model': 'ollama/llama3.1',
#     'max_tokens': 2048,
#     'stream': True,
#     'stop': None,
#     'api_base': 'http://localhost:11434',
#     'messages': [
#         {
#             'role': 'prompt'
#             'content': "<|im_start|>system\nYou are a friendly assistant. You answer questions from users.<|im_end|>\n<|im_start|>user\nAnswer the following question using only the context below. Only include information\nspecifically discussed.\n\nquestion: How do create agents with tasks?\ncontext: Tags: concepts, cli\n---\ntitle: CLI\ndescription: Learn how to use the CrewAI CLI to interact with CrewAI.\nicon: terminal\n---\n\n# CrewAI CLI Documentation\n\nThe CrewAI CLI provides a set of commands to interact with CrewAI, allowing you to create, train, run, and manage crews & flows.\n\n## Installation\n\nTo use the CrewAI CLI, make sure you have CrewAI installed:\n\n```shell\npip install crewai\n```\n\n## Basic Usage\n\nThe basic structure of a CrewAI CLI command is:\n\n```shell\ncrewai [COMMAND] [OPTIONS] [ARGUMENTS]\n```\n\n## Available Commands\n\n### 1. Create\n\nCreate a new crew or pipeline.\n\n```shell\ncrewai create [OPTIONS] TYPE NAME\n```\n\n- `TYPE`: Choose between \"crew\" or \"pipeline\"\n- `NAME`: Name of the crew or pipeline\n- `--router`: (Optional) Create a pipeline with router functionality\n\nExample:\n```she <|im_end|>\n<|im_start|>assistant\n",
#         }
#     ]
# }
response = completion(
    model="ollama/llama3.1",
    messages=[{"content": "respond in 20 words. who are you?", "role": "user"}],
    api_base="http://localhost:11434",
    stream=True
    # **params
)
print(response)
for chunk in response:
    delta = chunk['choices'][0]['delta']
    content = delta['content']
    logger.success(content, flush=True)
