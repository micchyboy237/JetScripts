import os
import shutil
import venv
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

work_dir = Path(f"{OUTPUT_DIR}/coding")
work_dir.mkdir(parents=True, exist_ok=True)

venv_dir = work_dir / ".venv"
venv_builder = venv.EnvBuilder(with_pip=True)
venv_builder.create(venv_dir)
venv_context = venv_builder.ensure_directories(venv_dir)


async def main():
    local_executor = LocalCommandLineCodeExecutor(
        work_dir=work_dir, virtual_env_context=venv_context)

    # Debug: List directory contents before execution with full paths, each on a new line
    logger.debug("Directory contents before execution:")
    logger.success(format_json(
        [str((work_dir / item).resolve()) for item in os.listdir(work_dir)]))

    # Save the code block to a persistent file for inspection with absolute path
    persistent_file = (work_dir / "executed_code.py").resolve()
    with open(persistent_file, "w") as f:
        f.write("print('Hello, World!')")

    result = await local_executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello, World!')"),
        ],
        cancellation_token=CancellationToken(),
    )

    # Save the execution output to a file with absolute path
    output_file = (work_dir / "execution_output.txt").resolve()
    with open(output_file, "w") as f:
        f.write(result.output)

    # Debug: List directory contents after execution with full paths, each on a new line
    logger.debug("Directory contents after execution:")
    logger.success(format_json(
        [str((work_dir / item).resolve()) for item in os.listdir(work_dir)]))

    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
