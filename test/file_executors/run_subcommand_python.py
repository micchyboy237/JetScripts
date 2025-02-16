from jet.executor.command import run_command
from jet.logger import logger
import json

file_to_execute = './execute_python_file.py'

model = "urchade/gliner_small-v2.1"
# model = "urchade/gliner_medium-v2.1"
style = "ent"

text = """
Role Overview:

We are seeking a skilled app developer to build our mobile app from scratch, integrating travel booking, relocation services, and community features. You will lead the design, development, and launch of our travel app.

Responsibilities:

Develop and maintain a scalable mobile app for iOS & Android

Integrate booking systems, payment gateways, and user profiles

Ensure seamless user experience & mobile responsiveness

Work with the team to test & refine the app before launch

Implement security features to protect user data

Qualifications:

3+ years of mobile app development (React Native, Flutter, Swift, or Kotlin)

Experience with APIs, databases, and cloud-based deployment

Strong UI/UX skills to create a user-friendly interface

Previous work on travel, booking, or e-commerce apps (preferred)

Ability to work independently & meet deadlines
"""

labels = ["role", "application", "technology stack", "qualifications"]

style = "ent"


def determine_chunk_size(text: str) -> int:
    """Dynamically set chunk size based on text length."""
    length = len(text)

    if length < 1000:
        return 250  # Small text, use smaller chunks
    elif length < 3000:
        return 350  # Medium text, moderate chunks
    else:
        return 500  # Large text, larger chunks


chunk_size = determine_chunk_size(text)

# Convert list to JSON string (to handle spaces safely)
labels = json.dumps(labels)
chunk_size = str(chunk_size)

# Construct command string
command_separator = "||"
command_args = [
    "python",
    file_to_execute,
    model,
    text,
    labels,
    style,
    chunk_size
]
command = command_separator.join(command_args)

# Use run_command instead of subprocess.run
logger.newline()
logger.info("Output:")

error_lines = []
debug_lines = []
success_lines = []

for line in run_command(command, separator=command_separator):
    if line.startswith('error: '):
        message = line[7:-2]
        error_lines.append(message)
        # logger.error(message)
    elif line.startswith('result: '):
        message = line[8:-2]
        success_lines.append(message)
        logger.success(message)

        # yield make_serializable(message)
    else:
        message = line[6:-2]
        debug_lines.append(message)
        # logger.debug(message)

if not success_lines:
    logger.debug("\n".join(debug_lines))
    logger.error("\n".join(error_lines))
