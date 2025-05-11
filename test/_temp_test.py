import re
import asyncio
import logging
from typing import List
from jet.logger import logger


def generate_unique_function_name(line: str) -> str:
    """Generate a unique function name based on line content."""
    import hashlib
    return f"async_func_{hashlib.md5(line.encode()).hexdigest()[:8]}"


def format_json(data):
    """Placeholder for JSON formatting."""
    return str(data)


def wrap_await_code_multiline_args(code: str) -> str:
    """Wrap lines containing 'await' or 'async with' in standalone async functions, handling multiline calls."""
    lines = code.splitlines()
    updated_lines = []
    line_idx = 0

    while line_idx < len(lines):
        line = lines[line_idx].rstrip()

        # Handle 'async with' statements
        if line.strip().startswith("async with"):
            leading_spaces = len(line) - len(line.lstrip())
            async_fn_name = f"async_func_{line_idx}"
            variable = "result"  # Use 'result' to match the variable inside the block

            # Collect the async with block
            async_block = [
                f"{' ' * leading_spaces}async def {async_fn_name}():"]
            async_block.append(f"{' ' * (leading_spaces + 4)}{line.strip()}")

            # Collect indented block under async with
            line_idx += 1
            while line_idx < len(lines):
                next_line = lines[line_idx].rstrip()
                next_leading_spaces = len(next_line) - len(next_line.lstrip())
                if next_leading_spaces <= leading_spaces and next_line.strip():
                    break
                # Calculate relative indentation within the async with block
                relative_indent = next_leading_spaces - \
                    (leading_spaces + 4)  # Relative to async with
                if relative_indent < 0:
                    relative_indent = 0  # Ensure no negative indentation
                # Base indentation: 4 (async def) + 4 (async with) + relative_indent
                adjusted_indent = ' ' * (leading_spaces + 8 + relative_indent)
                async_block.append(f"{adjusted_indent}{next_line.lstrip()}")
                line_idx += 1

            # Add return statement at function level (4 spaces from async def)
            async_block.append(
                f"{' ' * (leading_spaces + 4)}return {variable}")
            async_block.append("")  # Empty line for readability
            async_block.append(
                f"{' ' * leading_spaces}{variable} = asyncio.run({async_fn_name}())")
            async_block.append(
                f"{' ' * leading_spaces}logger.success(format_json({variable}))")

            updated_lines.extend(async_block)
            continue

        # Handle 'await' statements with multiline calls
        if "await" in line and line.strip().endswith("("):
            match = re.match(r'(.*?)\s*=\s*await', line)
            if match:
                variable = match.group(1).strip()
                leading_spaces = len(line) - len(line.lstrip())
                async_fn_name = f"async_func_{line_idx}"

                # Collect multiline call
                async_block = [
                    f"{' ' * leading_spaces}async def {async_fn_name}():"]
                async_block.append(
                    f"{' ' * (leading_spaces + 4)}{line.strip()}")

                open_parens = 1
                line_idx += 1
                while line_idx < len(lines) and open_parens > 0:
                    next_line = lines[line_idx].rstrip()
                    next_leading_spaces = len(
                        next_line) - len(next_line.lstrip())
                    relative_indent = next_leading_spaces - leading_spaces
                    if relative_indent < 0:
                        relative_indent = 0
                    adjusted_indent = ' ' * \
                        (leading_spaces + 4 + relative_indent)
                    async_block.append(
                        f"{adjusted_indent}{next_line.lstrip()}")
                    open_parens += next_line.count("(") - next_line.count(")")
                    line_idx += 1

                async_block.append(
                    f"{' ' * (leading_spaces + 4)}return {variable}")
                async_block.append(
                    f"{' ' * leading_spaces}{variable} = asyncio.run({async_fn_name}())")
                async_block.append(
                    f"{' ' * leading_spaces}logger.success(format_json({variable}))")

                updated_lines.extend(async_block)
            else:
                updated_lines.append(line)
                line_idx += 1
            continue

        # Handle single-line 'await' statements
        if "await" in line:
            match = re.match(r'(.*?)(?=\s*= await)', line)
            text_before_await = match.group(1).strip() if match else ""
            leading_spaces = len(line) - len(line.lstrip())
            async_fn_name = generate_unique_function_name(line)

            async_block = [
                f"{' ' * leading_spaces}async def {async_fn_name}():",
                f"{' ' * (leading_spaces + 4)}{line.strip()}",
                f"{' ' * (leading_spaces + 4)}return {text_before_await}",
                f"{' ' * leading_spaces}{text_before_await} = asyncio.run({async_fn_name}())",
                f"{' ' * leading_spaces}logger.success(format_json({text_before_await}))",
            ]

            updated_lines.extend(async_block)
            line_idx += 1
            continue

        # Non-await lines
        updated_lines.append(line)
        line_idx += 1

    return "\n".join(updated_lines)


class CodeBlock:
    def __init__(self, language: str, code: str):
        self.language = language
        self.code = code


class CancellationToken:
    pass


class DockerCommandLineCodeExecutor:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def execute_code_blocks(self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken):
        return "Execution Result"
