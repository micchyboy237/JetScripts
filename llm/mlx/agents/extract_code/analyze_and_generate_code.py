import os
import ast
import asyncio
import shutil
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import UserProxyAgent, AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from dotenv import load_dotenv

from jet.file.utils import save_file
from jet.llm.mlx.autogen_ext.mlx_chat_completion_client import MLXChatCompletionClient
from jet.models.model_types import LLMModelType

# Load environment variables
load_dotenv()

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Define the read_python_file function at module level


async def read_python_file(file_path: str) -> str:
    """Read the content of a Python file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


class CodeAnalyzer:
    """A reusable class to analyze Python files and process custom queries using AutoGen 0.4.3."""

    def __init__(self, model: LLMModelType = "mlx-community/Llama-3.2-3B-Instruct-4bit", work_dir="coding", timeout=60):
        """Initialize the CodeAnalyzer with model and executor settings."""
        self.model_client = MLXChatCompletionClient(model=model)
        self.work_dir = work_dir
        self.timeout = timeout
        self.executor = LocalCommandLineCodeExecutor(
            work_dir=self.work_dir,
            timeout=self.timeout,
        )
        self.assistant = None
        self.user_proxy = None
        self._setup_agents()

    def _setup_agents(self):
        """Set up the AssistantAgent and UserProxyAgent with AutoGen 0.4.3 syntax."""
        read_file_tool = FunctionTool(
            read_python_file,
            description="Read the content of a Python file given its path."
        )

        self.assistant = AssistantAgent(
            name="CodeAnalyzerAssistant",
            model_client=self.model_client,
            system_message=(
                "You are a coding assistant. Analyze Python files and process user queries. "
                "Identify relevant code (e.g., functions, classes, or logic) and provide clear, "
                "executable usage examples with explanations. Ensure accuracy and practicality."
            ),
            tools=[read_file_tool],
            reflect_on_tool_use=True,
        )

        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            description="A proxy for autonomous task execution and code analysis.",
        )

    def _parse_functions(self, code: str) -> list:
        """Parse Python code to extract function names and definitions."""
        try:
            tree = ast.parse(code)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    functions.append({
                        "name": node.name,
                        "code": "\n".join(code.splitlines()[start_line:end_line]),
                    })
            return functions
        except SyntaxError as e:
            return [{"error": f"Syntax error in code: {str(e)}"}]

    async def analyze_file(self, file_path: str, query: str) -> str:
        """Analyze a Python file and process a custom query."""
        if not os.path.exists(file_path):
            return f"Error: File {file_path} does not exist."

        # Read file content using the assistant's run method with the tool
        read_task = f"Use the read_python_file tool to read the content of {file_path}"
        read_result = await self.assistant.run(task=read_task)
        code = read_result.messages[-1].content if read_result.messages else f"Error: No content returned from read_python_file"
        if "Error reading file" in code:
            return code

        # Parse functions for reference
        functions = self._parse_functions(code)
        if not functions or "error" in functions[0]:
            return functions[0].get("error", "No functions found or parsing failed.")

        # Process the custom query
        full_query = (
            f"Analyze the following Python code:\n\n{code}\n\n"
            f"Query: {query}"
        )
        response = await self.user_proxy.run(task=full_query)
        return response.messages[-1].content if response.messages else "Error: No response from query processing"


# Example usage
if __name__ == "__main__":
    async def main():
        # Create a sample Python file with chunking logic
        sample_code = '''
def chunk_text(text: str, chunk_size: int) -> list:
    """Split text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_chunks(chunks: list) -> list:
    """Process each chunk (e.g., add prefix)."""
    return [f"Chunk_{i}: {chunk}" for i, chunk in enumerate(chunks)]
'''
        sample_file_path = "sample_code.py"
        os.makedirs("coding", exist_ok=True)
        with open(os.path.join("coding", sample_file_path), "w") as f:
            f.write(sample_code)
        save_file(sample_code, f"{OUTPUT_DIR}/coding/generated_code.py")

        # Initialize analyzer
        analyzer = CodeAnalyzer(work_dir=f"{OUTPUT_DIR}/work_dir")

        # Analyze the sample file with custom query
        query = "Extract the chunking logic with usage examples"
        result = await analyzer.analyze_file(os.path.join("coding", sample_file_path), query)
        print("Analysis Result:\n", result)

    # Run the example
    asyncio.run(main())
