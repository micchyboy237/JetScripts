import argparse
from pathlib import Path
import shutil

from jet.libs.smolagents.vision_web_browser import cli_args, main

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HEADLESS = True
DEFAULT_TASK = (
    "Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence "
    'containing the word "1992" that mentions a construction accident.'
)


if __name__ == "__main__":
    args = cli_args()

    # Resolve task: prefer -t/--task if given, otherwise use positional
    chosen_task = args.task_opt if args.task_opt is not None else args.task_pos

    main(
        headless=args.headless or DEFAULT_HEADLESS,
        task=chosen_task or DEFAULT_TASK,
        out_dir=args.out_dir or OUTPUT_DIR,
    )
