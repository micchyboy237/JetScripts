from jet.executor.command import run_command
from jet.file import load_file
from jet.file.utils import save_file
from jet.logger import logger
import json

data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/job-entities.json"

data = load_file(data_file)
texts = [
    f"Job Title: {item['title']}\n\n{item['details']}"
    if item['title'] not in item['details'].split("\n")[0] else item['details']
    for item in data
]


file_to_execute = './execute_python_file.py'

model = "urchade/gliner_small-v2.1"
# model = "urchade/gliner_medium-v2.1"
style = "ent"

# text = """
# Role Overview:

# We are seeking a skilled app developer to build our mobile app from scratch, integrating travel booking, relocation services, and community features. You will lead the design, development, and launch of our travel app.

# Responsibilities:

# Develop and maintain a scalable mobile app for iOS & Android

# Integrate booking systems, payment gateways, and user profiles

# Ensure seamless user experience & mobile responsiveness

# Work with the team to test & refine the app before launch

# Implement security features to protect user data

# Qualifications:

# 3+ years of mobile app development (React Native, Flutter, Swift, or Kotlin)

# Experience with APIs, databases, and cloud-based deployment

# Strong UI/UX skills to create a user-friendly interface

# Previous work on travel, booking, or e-commerce apps (preferred)

# Ability to work independently & meet deadlines
# """

labels = ["role", "application", "technology stack", "qualifications"]

style = "ent"


def determine_chunk_size(length: int) -> int:
    """Dynamically set chunk size based on text length."""

    if length < 1000:
        return 250  # Small text, use smaller chunks
    elif length < 3000:
        return 350  # Medium text, moderate chunks
    else:
        return 500  # Large text, larger chunks


max_text_length = max(len(text) for text in texts)
chunk_size = determine_chunk_size(max_text_length)

# Convert list to JSON string (to handle spaces safely)
texts = json.dumps(texts)
labels = json.dumps(labels)
chunk_size = str(chunk_size)

# Construct command string
command_separator = "<sep>"
command_args = [
    "python",
    file_to_execute,
    model,
    # text,
    texts,
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

entities = []

for line in run_command(command, separator=command_separator):
    if line.startswith('error: '):
        message = line[7:-2]
        error_lines.append(message)
        # logger.error(message)
    elif line.startswith('result: '):
        message = line[8:-2]
        success_lines.append(message)
        logger.success(message)

        result = json.loads(message)

        entities.append(result)

        save_file(entities, output_file)

        # yield make_serializable(message)
    else:
        message = line[6:-2]
        debug_lines.append(message)
        logger.debug(message)

if not success_lines:
    logger.error("\n".join(error_lines))
