from jet.executor.command import run_command
from jet.file import load_file
from jet.file.utils import save_file
from jet.logger import logger
import json

data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
file_to_execute = '/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/benchmark/execute_extract_job_entities.py'

output_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/job-entities.json"

data = load_file(data_file)
items = [
    {"id": item['id'], "text": f"Job Title: {item['title']}\n\n{item['details']}"}
    if item['title'] not in item['details'].split("\n")[0]
    else {"id": item['id'], "text": item['details']}
    for item in data
    if not item.get('entities')
]


model = "urchade/gliner_small-v2.1"
# model = "urchade/gliner_medium-v2.1"
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


# Convert list to JSON string (to handle spaces safely)
formatted_data = json.dumps(items)
labels = json.dumps(labels)

# Construct command string
command_separator = "<sep>"
command_args = [
    "python",
    file_to_execute,
    model,
    # text,
    formatted_data,
    labels,
    style,
]
command = command_separator.join(command_args)

# Use run_command instead of subprocess.run
logger.newline()
logger.info("Output:")

error_lines = []
debug_lines = []
success_lines = []

entities = []
entities_dict = {
    "model": model,
    "labels": labels,
    "results": entities,
}


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

        save_file(entities_dict, output_file)

        # Find the item with the matching 'id'
        data_item = next(
            (item for item in data if item["id"] == result["id"]), None)

        if data_item:
           # Initialize the dictionary to group by role
            data_item["entities"] = {}

            # Loop through the result entities
            for entity in result["entities"]:
                # Get the label for grouping (assuming it's the 'label' label or another key)
                label = entity['label']
                label = label.lower().replace(" ", "_")

                # If the label key does not exist in the dictionary, create it
                if label not in data_item["entities"]:
                    data_item["entities"][label] = []

                # Append the entity to the respective label group
                if entity['text'] not in data_item["entities"][label]:
                    data_item["entities"][label].append(entity['text'])

            # Save the updated data to the file
            save_file(data, data_file)

        # yield make_serializable(message)
    else:
        message = line[6:-2]
        debug_lines.append(message)
        logger.debug(message)

if not success_lines:
    logger.error("\n".join(error_lines))
