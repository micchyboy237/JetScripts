import os
import json
import random
import string
from aim import Run, Text


file_name = os.path.splitext(os.path.basename(__file__))[0]
repo_dir = f"./generated/{file_name}"

# Initialize a new run
run = Run(repo=repo_dir)

# Track string value
value = "Sample"
run.track("Sample", name='text')

#  Track dict value
value = {
    "prompt": "Sample prompt",
    "response": "Sample Value"
}
aim_text = Text(json.dumps(value, indent=2))
run.track(aim_text, name='text')
