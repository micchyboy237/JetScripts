import json
import os
import requests
from typing import Union, Generator, Optional
from jet.logger import logger
from jet.code import MarkdownCodeExtractor


class ChatAPI:
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434/api/chat"):
        self.model = model
        self.base_url = base_url

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        stream: bool = False,
        options: dict = {}
    ) -> Union[dict, Generator[str, None, None]]:
        """Handles the chat API logic with optional streaming."""
        stream = options.get("stream", stream)
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": options
        }

        logger.log("URL:", self.base_url, colors=["GRAY", "INFO"])
        logger.log(f"Params:")
        for payload_key, payload_value in payload.items():
            if isinstance(payload_value, dict) or isinstance(payload_value, list):
                logger.log(f"{payload_key}:", json.dumps(
                    payload_value, indent=2), colors=["GRAY", "INFO"])
            else:
                logger.log(f"{payload_key}:", payload_value,
                           colors=["GRAY", "INFO"])

        if stream:
            return self._stream_response(self.base_url, payload)
        else:
            return self._non_stream_response(self.base_url, payload)

    def _stream_response(self, url: str, payload: dict, headers: dict = {"Content-Type": "application/json"}) -> Generator[str, None, None]:
        logger.log("_stream_response")
        try:
            # response = requests.post(url, json=payload, stream=True)
            with requests.post(url, headers=headers, json=payload, stream=payload['stream']) as response:
                if payload['stream']:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if line:
                            data_str = line.decode("utf-8")

                            # Load the line as JSON
                            try:
                                decoded_json = json.loads(data_str)
                                yield decoded_json
                            except json.JSONDecodeError:
                                print(f"JSON error on chunk: {data_str}")
                else:
                    return response.json()

        except requests.exceptions.RequestException as e:
            yield f"Stream Error: {e}"
        except Exception as e:
            yield f"Unexpected Stream Error: {e}"

    def _non_stream_response(self, url: str, payload: dict) -> dict:
        try:
            response = requests.post(url, json=payload, timeout=(3.05, 60))
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {"error": f"Request Error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected Error: {e}"}


# @time_it
def count_tokens(model: str, text: str) -> int:
    """Counts tokens for a given text using the specified model."""
    import litellm
    return litellm.token_counter(model=model, text=text)


def serialize_messages(messages: list[dict]) -> str:
    """Serializes messages to a single text block for token counting."""
    # return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    return str(messages)

# Function to format duration in milliseconds to a readable string


def format_duration(ms):
    seconds = ms / 1000
    minutes = seconds / 60
    hours = minutes / 60

    if hours >= 1:
        return f"{hours:.2f} hours"
    elif minutes >= 1:
        return f"{minutes:.2f} minutes"
    elif seconds >= 1:
        return f"{seconds:.2f} seconds"
    else:
        # If the duration is less than a second, return in milliseconds
        return f"{ms} ms"


def save_file(data: str | dict | list, output_file: str = "generated/chat_result.json"):
    import os
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to file
    try:
        if output_file.endswith(".json"):
            if isinstance(data, str):
                data = json.loads(data)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.log(
                "Save JSON data to:",
                output_file,
                colors=["LOG", "BRIGHT_SUCCESS"]
            )
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(data)
            logger.log(
                "Save data to:",
                output_file,
                colors=["LOG", "BRIGHT_SUCCESS"]
            )
    except Exception as e:
        logger.error(f"Failed to save file: {e}")


def load_file(file_path: str):
    # Ensure the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    try:
        if file_path.endswith(".json"):
            # Load JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Read file as plain text
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()
        return data
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return None


# Example usage
if __name__ == "__main__":
    prompts_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/prompts"
    output_dir = os.path.join(prompts_dir,  "generated")
    all_results_output_file = os.path.join(
        output_dir, f"results.json")

    system_message_file = os.path.join(prompts_dir,  "React_JSON_System.md")

    # Context Prompts
    context_dir = os.path.join(prompts_dir, "context")
    context_output_dir = os.path.join(context_dir, "generated")

    initial_context_filename = "01_Initial_Requirements.md"
    file = os.path.join(context_output_dir, initial_context_filename)
    initial_requirements = load_file(file).strip()

    # Preparation Prompts
    prepare_dir = os.path.join(prompts_dir, "prepare")

    filename = "Prepare_DB_Prisma_Schema.md"
    file = os.path.join(prepare_dir, filename)
    contents = load_file(file).strip()
    final_prompt = contents.format(
        context=initial_requirements)
    response_output_file = os.path.join(
        output_dir, f"schema.prisma")

    # filename = "Prepare_JSON_Schema.md"
    # file = os.path.join(prepare_dir, filename)
    # contents = load_file(file).strip()
    # final_prompt = contents.format(
    #     context=initial_requirements)
    # response_output_file = os.path.join(
    #     output_dir, f"schema.json")

    # filename = "Prepare_Requirements.md"
    # file = os.path.join(prepare_dir, filename)
    # contents = load_file(file).strip()
    # final_prompt = contents.format(
    #     context=initial_requirements)
    # response_output_file = os.path.join(
    #     output_dir, f"requirements.md")

    # filename_no_ext = "02_Write_Documentation"
    # file = os.path.join(context_dir, filename_no_ext)
    # contents = load_file(file).strip()
    # write_documentation_prompt = contents.format(
    #     requirements=initial_requirements)
    # contents_output_file = os.path.join(
    #     context_output_dir, f"{filename_no_ext}.md")
    # info_output_file = os.path.join(
    #     context_output_dir, f"{filename_no_ext}.json")

    # Eval Prompts
    eval_dir = os.path.join(prompts_dir, "eval")
    eval_output_dir = os.path.join(eval_dir, "generated")

    text_to_schema_file = os.path.join(
        eval_dir, "01_React_Text_To_Schema.prompt.md")
    generated_schema_file = os.path.join(eval_output_dir, "schema.json")

    schema_to_code_file = os.path.join(
        eval_dir, "02_React_Schema_To_Code.prompt.md")
    generated_code_file = os.path.join(eval_output_dir, "evaluated-react.js")

    code_eval_file = os.path.join(
        eval_dir, "03_React_Code_Evaluation.prompt.md")
    generated_eval_file = os.path.join(eval_output_dir, "evaluation.json")

    # prompt_template = load_file(prompt_template_path).strip()
    system_message = load_file(system_message_file).strip()
    text_to_schema_prompt = load_file(text_to_schema_file).strip()
    schema_to_code_prompt = load_file(schema_to_code_file).strip()
    code_eval_prompt = load_file(code_eval_file).strip()

    with open(generated_schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)
        schema = json.dumps(schema, indent=2)

    with open(generated_code_file, "r", encoding="utf-8") as f:
        code = f.read()
        language = "javascript"

    # final_prompt = write_documentation_prompt
    # final_prompt = text_to_schema_prompt
    # final_prompt = schema_to_code_prompt.format(schema=schema)
    # final_prompt = code_eval_prompt.format(
    #     language=language, code=code, schema=schema)
    logger.log("PROMPT:")
    logger.debug(final_prompt)

    model = "codellama"
    # model = "qwen2.5-coder"
    base_url = "http://localhost:11434/api/chat"
    options = {
        "seed": -1,
        "stream": True,
        "num_batch": 512,
        "num_thread": 4,
        "temperature": 0.7,
        "num_ctx": 4096,
        "num_predict": -1,
        "use_mmap": True,
        "use_mlock": False,
        "num_gpu": 0,
        "num_keep": 0,
    }
    chat_api = ChatAPI(model, base_url)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": final_prompt}
    ]

    logger.log("Generating response...")

    # Updated token counting
    raw_prompt_tokens = count_tokens(model=model, text=final_prompt)
    serialized_messages = serialize_messages(messages)
    messages_tokens = count_tokens(model=model, text=serialized_messages)

    logger.log("raw_prompt_tokens:", raw_prompt_tokens,
               colors=["GRAY", "DEBUG"])
    logger.log("messages_tokens:", messages_tokens, colors=["GRAY", "DEBUG"])

    output = ""
    metadata = {
        "input": {
            "raw_prompt_tokens": raw_prompt_tokens,
            "messages_tokens": messages_tokens,
        },
        "output": {}
    }

    for chunk_json in chat_api.chat(messages, options=options):
        content = chunk_json['message']['content']
        logger.success(content, flush=True)
        output += content

        if chunk_json['done']:
            # Extract metadata from the final chunk
            metadata["output"] = {
                "total_duration": format_duration(chunk_json['total_duration']),
                "load_duration": format_duration(chunk_json['load_duration']),
                "prompt_eval_count": chunk_json['prompt_eval_count'],
                "prompt_eval_duration": format_duration(chunk_json['prompt_eval_duration']),
                "eval_count": chunk_json['eval_count'],
                "eval_duration": format_duration(chunk_json['eval_duration'])
            }

    print("\n")

    logger.log("Summary:")
    logger.debug(json.dumps(metadata, indent=2))

    print("\n")

    # Save response
    extractor = MarkdownCodeExtractor()
    code_blocks = extractor.extract_code_blocks(output)
    save_file(code_blocks[0]['code'], response_output_file)

    # Save summary
    summary = {
        "filename": filename,
        "base_url": base_url,
        "model": model,
        "options": options,
        "prompt": final_prompt,
        "response": output,
        "metadata": metadata
    }
    all_results = load_file(all_results_output_file) or []
    save_file(all_results + [summary], all_results_output_file)
