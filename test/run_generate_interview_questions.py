import json
import os

from jet.llm.llm_types import Message, MessageRole
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.utils.markdown import extract_json_block_content
from jet.validation.main.json_validation import validate_json
from tqdm import tqdm
from jet.llm.ollama.base import Ollama, initialize_ollama_settings
from jet.logger import logger
from jet.file import save_file
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.base.llms.types import ChatMessage
initialize_ollama_settings()

llm = Ollama(model="llama3.1")

data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
sub_dir = "jet-resume-summary"

documents = SimpleDirectoryReader(
    data_dir, recursive=True).load_data()
generated_dir = os.path.join(
    "generated", os.path.basename(__file__).split('.')[0], sub_dir)
os.makedirs(generated_dir, exist_ok=True)

SYSTEM_MESSAGE = """
You are an AI assistant that follows instructions. You can generate autocompletion prompts for users, providing them with relevant options to complete their sentences or phrases.
You will respond by outputting a single JSON block surrounded by ```json containing a list of strings in the following format: [string1, string2, ...].
""".strip()

USER_MESSAGE_TEMPLATE = "You are an employer who wants to generate questions for a job interview\nContext:\n{context_str}"


# Generate summary
generation_tqdm = tqdm(documents, total=len(documents))
for tqdm_idx, doc in enumerate(generation_tqdm):
    file_name = doc.metadata['file_name']
    file_path = doc.metadata['file_path']

    title = file_name.split(".")[0]

    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    user_message = USER_MESSAGE_TEMPLATE.format(context_str=file_content)
    messages: list[ChatMessage] = [
        ChatMessage(content=SYSTEM_MESSAGE, role=MessageRole.SYSTEM),
        ChatMessage(content=user_message, role=MessageRole.USER),
    ]
    response = llm.chat(messages)

    extracted_result = extract_json_block_content(
        response.message.content or "")
    validation_result = validate_json(extracted_result)
    response.message.content = json.dumps(validation_result['data'])

    output_file = os.path.join(generated_dir, f"{title}.json")

    messages.append(response.message)

    results = [Message(role=item.role.value, content=item.content,
                       **item.additional_kwargs) for item in messages]

    save_file(results, output_file)

    generation_tqdm.update(1)
