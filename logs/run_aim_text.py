import os
import random
import string
from aim import Run, Text


file_name = os.path.splitext(os.path.basename(__file__))[0]
repo_dir = f"./generated/{file_name}"

# Initialize a new run
run = Run(repo=repo_dir)

for step in range(100):
    # Generate a random string for this example
    random_str = ''.join(random.choices(
        string.ascii_uppercase +
        string.digits, k=20)
    )
    aim_text = Text(random_str)
    run.track(aim_text, name='text', step=step)
