import asyncio
import os
import subprocess
import sys
import json
import logging
import importlib.util
from typing import Sequence
from autogen_core import CancellationToken
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response
from mlx_lm import load, generate
import esprima

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logger
logger = logging.getLogger("minified_converter")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Auto-create and setup .venv
venv_dir = os.path.join(os.getcwd(), ".venv")
venv_python = os.path.join(venv_dir, "bin", "python") if sys.platform != "win32" else os.path.join(
    venv_dir, "Scripts", "python.exe")


def create_venv():
    if not os.path.exists(venv_dir):
        logger.info(f"Creating virtual environment in {venv_dir}...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
            logger.info("Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            sys.exit(1)


def install_python_package(package: str):
    try:
        subprocess.check_call([venv_python, "-m", "pip", "install", package])
        logger.info(f"Installed Python package: {package}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")
        sys.exit(1)


def check_python_dependencies(packages: list[str]):
    for package in packages:
        spec = importlib.util.find_spec(package.replace("-", "_"))
        if spec is None:
            logger.info(f"Package {package} not found, installing in .venv...")
            install_python_package(package)


def install_node_package(package: str):
    try:
        subprocess.check_call(["npm", "install", package])
        logger.info(f"Installed Node.js package: {package}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")
        sys.exit(1)


def check_node_dependencies(packages: list[str]):
    for package in packages:
        try:
            subprocess.check_call(
                ["npm", "list", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logger.info(f"Package {package} not found, installing...")
            install_node_package(package)


# Setup environment
create_venv()
check_python_dependencies(["mlx-lm", "tokenizers", "esprima"])
check_node_dependencies(["prettier@3.5.3", "esprima"])

# Rest of the script (unchanged from previous response)


class MLXChatCompletionClient:
    def __init__(self, model_name: str, max_context_length: int = 64000):
        self.model, self.tokenizer = load(model_name)
        self.max_context_length = max_context_length
        self.model_info = {"function_calling": False}

    async def create(self, messages: Sequence[BaseChatMessage], max_tokens: int = 3000) -> str:
        prompt = ""
        for msg in messages:
            if isinstance(msg, TextMessage):
                prompt += f"{msg.source}: {msg.content}\n"

        input_tokens = len(self.tokenizer.encode(prompt))
        if input_tokens >= self.max_context_length - max_tokens:
            raise ValueError("Input exceeds context length limit.")

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=min(max_tokens, self.max_context_length - input_tokens),
            verbose=True,
        )
        return response.strip()

    async def close(self):
        pass


def chunk_code(code: str, tokenizer, max_tokens: int = 10000) -> list[str]:
    tokens = tokenizer.encode(code)
    if len(tokens) <= max_tokens:
        return [code]
    chunks, current_chunk = [], []
    current_length = 0
    for token in tokens:
        if current_length + 1 > max_tokens:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = [token]
            current_length = 1
        else:
            current_chunk.append(token)
            current_length += 1
    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))
    return chunks


def validate_js(code: str) -> dict:
    try:
        esprima.parseScript(code)
        return {"is_valid": True, "message": "Valid JavaScript syntax"}
    except Exception as e:
        return {"is_valid": False, "message": f"Syntax error: {str(e)}"}


def run_prettier(code: str, output_file: str = None) -> str:
    temp_file = output_file if output_file else "temp.js"
    with open(temp_file, "w") as f:
        f.write(code)
    try:
        result = subprocess.run(
            ["npx", "prettier", "--write", temp_file],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Prettier output: {result.stdout}")
        with open(temp_file, "r") as f:
            formatted_code = f.read()
        if not output_file:
            os.remove(temp_file)
        return formatted_code
    except subprocess.CalledProcessError as e:
        logger.error(f"Prettier failed: {e.stderr}")
        if not output_file:
            os.remove(temp_file) if os.path.exists(temp_file) else None
        return code


class FormatterAgent(AssistantAgent):
    def __init__(self, name: str, model_client):
        super().__init__(
            name=name,
            system_message=(
                "You are an expert JavaScript developer. Convert minified JavaScript to readable, "
                "well-formatted JavaScript with proper indentation (2 spaces), consistent spacing, and "
                "meaningful variable names. Rename obfuscated variables (e.g., Z.__v, Z.__k) to descriptive "
                "names (e.g., virtualDom, keyFunction). Output only valid JavaScript code, no explanations or comments."
            ),
            model_client=model_client,
            model_context=BufferedChatCompletionContext(buffer_size=10),
        )
        self.model_client = model_client

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if not messages:
            return Response(chat_message=TextMessage(content="No code provided.", source=self.name))

        last_message = messages[-1]
        if not isinstance(last_message, TextMessage):
            return Response(chat_message=TextMessage(content="Invalid input: Expected text message with JavaScript code.", source=self.name))

        try:
            formatted_code = await self.model_client.create([TextMessage(content=last_message.content, source="system")])
            prettier_result = run_prettier(formatted_code)
            self.model_context.add_message(TextMessage(
                content=prettier_result, source=self.name))
            return Response(chat_message=TextMessage(content=prettier_result, source=self.name))
        except Exception as e:
            error_message = f"Error formatting code: {str(e)}"
            logger.error(error_message)
            self.model_context.add_message(TextMessage(
                content=error_message, source=self.name))
            return Response(chat_message=TextMessage(content=error_message, source=self.name))


class ValidatorAgent(BaseChatAgent):
    def __init__(self, name: str, description: str):
        super().__init__(name=name, description=description)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if not messages:
            return Response(chat_message=TextMessage(content="No code to validate.", source=self.name))

        last_message = messages[-1]
        if not isinstance(last_message, TextMessage):
            return Response(chat_message=TextMessage(content="Invalid input: Expected text message with JavaScript code.", source=self.name))

        code = last_message.content
        validation_result = validate_js(code)
        return Response(chat_message=TextMessage(content=str(validation_result), source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)


async def convert_minified_js(input_file: str, output_file: str) -> bool:
    model_client = MLXChatCompletionClient(
        "mlx-community/Llama-3.2-3B-Instruct-4bit")
    formatter = FormatterAgent(
        name="FormatterAgent", model_client=model_client)
    validator = ValidatorAgent(
        name="ValidatorAgent", description="Validates JavaScript syntax using Esprima")

    # Initialize context file
    context_file = "chunk_context.json"
    context_data = {"chunks": []}
    formatted_chunks = []

    # Load existing context if available
    if os.path.exists(context_file):
        with open(context_file, "r") as f:
            context_data = json.load(f)
        formatted_chunks = [c["formatted"]
                            for c in context_data["chunks"] if c["valid"]]
        logger.info(
            f"Loaded {len(formatted_chunks)} valid chunks from context")

    logger.info(f"📂 Reading minified code from: {input_file}")
    with open(input_file, "r") as f:
        minified_code = f.read()

    # Validate input file
    if not validate_js(minified_code)["is_valid"]:
        logger.error("Invalid input JavaScript")
        await model_client.close()
        return False

    logger.info("✂️ Chunking code for processing...")
    chunks = chunk_code(minified_code, model_client.tokenizer)
    logger.info(f"✅ Total chunks created: {len(chunks)}")

    # Skip already processed chunks
    processed_indices = [c["index"]
                         for c in context_data["chunks"] if c["valid"]]
    chunks_to_process = [c for i, c in enumerate(
        chunks, 1) if i not in processed_indices]
    has_errors = False
    previous_output = formatted_chunks[-1] if formatted_chunks else ""

    for i, chunk in enumerate(chunks_to_process, len(formatted_chunks) + 1):
        logger.info(f"🧩 Chunk {i}/{len(chunks)}: Starting formatting...")
        try:
            # Include previous output for context
            prompt = (
                f"Previous formatted code:\n```javascript\n{previous_output}\n```\n"
                f"Convert this minified JavaScript to readable, well-formatted JavaScript, continuing from the previous context:\n"
                f"```javascript\n{chunk}\n```"
            )
            formatter_response = await formatter.on_messages(
                [TextMessage(content=prompt, source="user")
                 ], CancellationToken()
            )
            if not isinstance(formatter_response.chat_message, TextMessage):
                logger.error(
                    f"❌ Formatting failed for chunk {i}: Invalid response")
                has_errors = True
                context_data["chunks"].append(
                    {"index": i, "formatted": "", "valid": False, "error": "Invalid response"})
                with open(context_file, "w") as f:
                    json.dump(context_data, f)
                continue

            formatted_code = formatter_response.chat_message.content
            logger.info(
                f"✅ Chunk {i}: Formatting complete. Validating syntax...")

            validator_response = await validator.on_messages(
                [TextMessage(content=formatted_code,
                             source="formatter")], CancellationToken()
            )
            if not isinstance(validator_response.chat_message, TextMessage):
                logger.error(
                    f"❌ Validation failed for chunk {i}: Invalid response")
                has_errors = True
                context_data["chunks"].append(
                    {"index": i, "formatted": "", "valid": False, "error": "Invalid validation response"})
                with open(context_file, "w") as f:
                    json.dump(context_data, f)
                continue

            validation_result = eval(validator_response.chat_message.content)
            if not validation_result["is_valid"]:
                logger.error(
                    f"❌ Syntax error in chunk {i}: {validation_result['message']}")
                has_errors = True
                context_data["chunks"].append(
                    {"index": i, "formatted": formatted_code, "valid": False, "error": validation_result['message']})
                with open(context_file, "w") as f:
                    json.dump(context_data, f)
                continue

            logger.info(f"✅ Chunk {i}: Passed validation")
            formatted_chunks.append(formatted_code)
            previous_output = formatted_code
            context_data["chunks"].append(
                {"index": i, "formatted": formatted_code, "valid": True})
            with open(context_file, "w") as f:
                json.dump(context_data, f)

        except Exception as e:
            logger.error(f"❌ Error processing chunk {i}: {str(e)}")
            has_errors = True
            context_data["chunks"].append(
                {"index": i, "formatted": "", "valid": False, "error": str(e)})
            with open(context_file, "w") as f:
                json.dump(context_data, f)
            continue

    if not formatted_chunks:
        logger.error("❌ No chunks were successfully processed")
        await model_client.close()
        return False

    logger.info("🎉 All chunks processed successfully")
    logger.info("🧵 Combining and formatting all chunks...")
    final_code = "\n".join(formatted_chunks)
    final_code = run_prettier(final_code, output_file=output_file)

    logger.info(f"🔎 Running final validation on output file: {output_file}")
    final_validation = validate_js(final_code)
    await model_client.close()

    if final_validation["is_valid"]:
        logger.info(
            f"🎉 Success: Minified code converted to readable format → {output_file}")
        return True
    else:
        logger.error(
            f"❌ Final validation failed: {final_validation['message']}")
        return not has_errors

if __name__ == "__main__":
    input_file = "/Users/jethroestrada/Desktop/External_Projects/AI/rag_05_2025/docsearch/_jet_sample/index.min.js"
    output_file = "readable.js"
    asyncio.run(convert_minified_js(input_file, output_file))
