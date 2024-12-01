"""
title: Run File Manager
description: Auto creates files based on discussed project structure.
"""

import os
import re
import json
import datetime
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Union, Optional, TypedDict, Literal
from fastapi.requests import Request
from subprocess import Popen, PIPE

DEFAULT_GENERATED_DIR = "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/open-webui/generated"

# ANSI color codes
BOLD = "\u001b[1m"
RED = BOLD + "\u001b[38;5;196m"
GREEN = BOLD + "\u001b[38;5;40m"
BRIGHT_GREEN = BOLD + "\u001b[48;5;40m"
RESET = "\u001b[0m"


def colorize(text: str, color: str):
    return f"{color}{text}{RESET}"


class UserMessageTextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class UserMessageImageContentPart(TypedDict):
    type: Literal["image_url"]
    # Ignoring the other "detail param for now"
    image_url: Dict[Literal["url"], str]


CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\n(.*?)\n```"
FILE_PATH_PATTERN = r".*File Path: `?([^`]+)`?"
UNKNOWN = "unknown"

LANGUAGE_EXTENSION_MAP = {
    "bash": "sh",
    "python": "py",
    "javascript": "js",
    "typescript": "ts",
    "markdown": "md",
    "jsx": "jsx",
    "tsx": "tsx",
    "json": "json",
    "unknown": "txt"  # Default fallback
}


def content_str(content: Union[str, List[Union[UserMessageTextContentPart, UserMessageImageContentPart]], None]) -> str:
    """Converts the `content` field of an OpenAI message into a string format.

    This function processes content that may be a string, a list of mixed text and image URLs, or None,
    and converts it into a string. Text is directly appended to the result string, while image URLs are
    represented by a placeholder image token. If the content is None, an empty string is returned.

    Args:
        - content (Union[str, List, None]): The content to be processed. Can be a string, a list of dictionaries
                                      representing text and image URLs, or None.

    Returns:
        str: A string representation of the input content. Image URLs are replaced with an image token.

    Note:
    - The function expects each dictionary in the list to have a "type" key that is either "text" or "image_url".
      For "text" type, the "text" key's value is appended to the result. For "image_url", an image token is appended.
    - This function is useful for handling content that may include both text and image references, especially
      in contexts where images need to be represented as placeholders.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(
            f"content must be None, str, or list, but got {type(content)}")

    rst = ""
    for item in content:
        if not isinstance(item, dict):
            raise TypeError(
                "Wrong content format: every element should be dict if the content is a list.")
        assert "type" in item, "Wrong content format. Missing 'type' key in content's dict."
        if item["type"] == "text":
            rst += item["text"]
        elif item["type"] == "image_url":
            rst += "<image>"
        else:
            raise ValueError(
                f"Wrong content format: unknown type {item['type']} within the content")
    return rst


def infer_lang(code: str) -> str:
    """infer the language for the code.
    TODO: make it robust.
    """
    if code.startswith("python ") or code.startswith("pip") or code.startswith("python3 "):
        return "sh"

    # check if code is a valid python code
    try:
        compile(code, "test", "exec")
        return "python"
    except SyntaxError:
        # not a valid python code
        return UNKNOWN


class CodeBlock(TypedDict):
    """Represents a code block with associated metadata."""
    code: str  # The code content.
    language: str  # Programming language of the code block.
    file_path: Optional[str]  # The file path associated with the code block.


class MarkdownCodeExtractor:
    """Extracts code blocks from Markdown content."""

    def extract_code_blocks(self, markdown: str) -> List[CodeBlock]:
        """Extract code blocks and associated file paths from Markdown text.

        Args:
            markdown (str): Markdown content.

        Returns:
            List[CodeBlock]: List of extracted code blocks.
        """
        lines = markdown.strip().splitlines()
        code_blocks: List[CodeBlock] = []
        current_file_path: Optional[str] = None
        inside_code_block = False
        lang = None
        code_lines = []

        for line in lines:
            # Check for file path pattern
            file_path_match = re.match(FILE_PATH_PATTERN, line)
            if file_path_match:
                current_file_path = file_path_match.group(1)
                continue

            # Handle start of a code block
            if line.startswith("```"):
                if not inside_code_block:
                    inside_code_block = True
                    lang = line.strip("`").strip() or UNKNOWN
                    code_lines = []
                else:
                    # End of a code block
                    code_content = "\n".join(code_lines).rstrip()
                    if code_content:
                        code_blocks.append(
                            CodeBlock(
                                code=code_content,
                                language=lang,
                                file_path=current_file_path
                            )
                        )
                    inside_code_block = False
                    current_file_path = None
                continue

            # Collect lines inside a code block
            if inside_code_block:
                code_lines.append(line)

        return code_blocks


class Action:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def save_file(self, code: str, path: str, lang: str = None, base_dir: str = "generated") -> dict:
        """Saves code to the specified file path, appending the language as the file extension.

        Args:
            code (str): The code to save.
            path (str): The base file path.
            lang (str): The programming language of the code (used as the file extension).
            base_dir (str): The base directory containing the saved file

        Returns:
            dict: A dictionary containing the success status and optional error message.
        """
        try:
            # Add the language as the file extension
            splitted_paths = path.split('.')
            curr_ext = splitted_paths[-1] if len(splitted_paths) > 1 else lang
            lang = LANGUAGE_EXTENSION_MAP.get(lang, curr_ext)
            file_ext = f".{lang}" if lang else ""
            if lang and path[-len(file_ext):] != file_ext:
                path = f"{path}{file_ext}"

            path = os.path.join(base_dir, path)

            # Ensure the directory exists
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Write the code to the file
            with open(path, 'w', encoding='utf-8') as file:
                file.write(code)
            print(f"File saved at {colorize(path, BRIGHT_GREEN)}")

            return {"success": True, "message": f"File saved at {path}", "file": file, "lang": lang, "code": code}
        except Exception as e:
            return {"success": False, "message": str(e), "file": file, "lang": lang, "code": code}

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        print(f"action:{__name__}")

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        if __event_emitter__:
            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Processing your input", "done": False},
                    }
                )

            last_user_message = body["messages"][-2]
            last_assistant_message = body["messages"][-1]

            prompt_text = last_user_message["content"]
            response_text = last_assistant_message["content"]

            # Find all code blocks from last AI response
            extractor = MarkdownCodeExtractor()
            code_blocks: list[CodeBlock] = extractor.extract_code_blocks(
                response_text)
            # Filter code_blocks
            failed_code_blocks = []
            filtered_code_blocks = []
            for code in code_blocks:
                valid_code_block = code['code'] and code['language'] and code['file_path']
                if valid_code_block:
                    filtered_code_blocks.append(code)
                else:
                    failed_code_blocks.append(code)
            code_blocks = filtered_code_blocks

            sub_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            generated_dir = os.getenv("GENERATED_DIR", DEFAULT_GENERATED_DIR)
            base_dir = os.path.join(generated_dir, sub_dir)
            file_results = []
            for code in code_blocks:
                result = self.save_file(
                    code['code'], code['file_path'], code['language'], base_dir=base_dir)
                file_results.append(result)

            meta_data = {
                "prompt_text": prompt_text,
                "response_text": response_text,
                "code_blocks": code_blocks,
                "failed_code_blocks": failed_code_blocks,
            }

            if code_blocks:
                result = self.save_file(
                    json.dumps(meta_data, indent=2, ensure_ascii=False), "meta.json", "json", base_dir=base_dir)
                file_results.append(result)

            if file_results:
                return {"type": "code_execution_result", "data": {"output": file_results, "done": True}}

            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "No valid code block detected",
                            "done": True,
                        },
                    }
                )


if __name__ == "__main__":
    async def main():
        # Sample usage demonstrating the file manager
        sample_body = {
            "messages": [
                # User message
                {"content": "Please create a Python script, HTML file, and shell script"},
                {"content": '''
File Path: `src/hello.py`
```python
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
```

File Path: `src/index.html`
```html
<!DOCTYPE html>
<html>
<head>
    <title>Sample Page</title>
</head>
<body>
    <h1>Welcome</h1>
    <p>This is a sample page.</p>
</body>
</html>
```

File Path: `src/setup.sh`
```bash
#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/logs
mkdir -p data/output

# Set up environment variables
export APP_ENV="development"
export APP_PORT=8000

echo "Setup complete! Starting server..."
python src/hello.py
```
'''}  # Assistant message
            ]
        }

        # Initialize action
        action = Action()

        # Create mock event emitter
        async def mock_event_emitter(event):
            print(f"Event emitted: {event}")

        # Create mock user with default valves
        mock_user = {"valves": action.UserValves()}

        # Execute action
        result = await action.action(
            body=sample_body,
            __user__=mock_user,
            __event_emitter__=mock_event_emitter,
            __event_call__=None
        )

        if result:
            print("\nFiles generated successfully:")
            for file_result in result["data"]["output"]:
                if file_result["success"]:
                    print(f"{colorize('✓', GREEN)} {file_result['file'].name}")
                else:
                    print(
                        f"{colorize('✗', RED)} Failed to create file: {file_result['message']}")

    # Run the async main function
    import asyncio
    asyncio.run(main())
