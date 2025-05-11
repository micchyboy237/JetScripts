import subprocess
from pathlib import Path
from typing import List
import sys
import os

from jet.logger import CustomLogger


def find_python_scripts(base_dir: str, exclude_self: bool = True) -> List[Path]:
    base_path = Path(base_dir).resolve()
    all_scripts = [
        p for p in base_path.rglob("*.py")
        if p.name != "__init__.py"
    ]
    if exclude_self:
        self_path = Path(__file__).resolve()
        all_scripts = [p for p in all_scripts if p != self_path]
    return all_scripts


def run_script(script_path: Path) -> subprocess.CompletedProcess:
    script_dir = os.path.dirname(script_path)
    log_file = os.path.join(
        script_dir, f"{os.path.splitext(os.path.basename(script_path))[0]}.log")
    logger = CustomLogger(log_file, overwrite=True)

    logger.info(f"Running: {script_path}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    logger.debug(f"Output ({script_path}):\n{result.stdout}")
    if result.stderr:
        logger.error(
            f"Error ({script_path}):\n{result.stderr}", file=sys.stderr)
    return result


def run_all_scripts(base_dir: str):
    scripts = find_python_scripts(base_dir)
    for script in scripts:
        run_script(script)


if __name__ == "__main__":
    # base_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    base_directory = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/mlx/helpers"
    run_all_scripts(base_directory)
