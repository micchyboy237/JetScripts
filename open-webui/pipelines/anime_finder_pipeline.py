from typing import List, Union, Generator, Iterator
import requests
import json
from jet.logger import logger
from jet._token import token_counter

DEFAULT_SYSTEM_MESSAGE = "You are a versatile language model capable of understanding and processing a wide variety of information, including structured and unstructured content like web pages, documents, or databases. Your primary role is to provide accurate, helpful, and easy-to-understand responses on topics related to anime when users ask questions or make requests. Strive to present your answers in a clear, friendly, and supportive manner while considering the context of the user's request."


class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.id = "anime-finder-pipeline"
        self.name = "Anime Finder Pipeline"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        OLLAMA_BASE_URL = "http://localhost:11434"
        URL = f"{OLLAMA_BASE_URL}/api/chat"
        MODEL = "llama3.1:latest"
        OPTIONS = {
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

        system_message = next(
            (message for message in messages if message["role"] == "system"), None)
        if not system_message:
            system_message = DEFAULT_SYSTEM_MESSAGE
            # Add system message
            body['messages'] = [
                {
                    "role": "system",
                    "content": system_message
                }
            ] + messages

        print("----- START CONTEXT -----")
        logger.log(f"SYSTEM MESSAGE:")
        logger.debug(system_message)
        logger.log(f"USER MESSAGE:")
        logger.debug(user_message)
        print("----- END CONTEXT -----")
        print("----- START MESSAGES -----")
        logger.log(f"MESSAGES ({len(body['messages'])}):")
        logger.debug(json.dumps(body['messages'], indent=2))
        print("----- END MESSAGES -----")
        print("----- START INFO -----")
        if "user" in body:
            logger.log("USER:")
            logger.info(
                f'{body["user"]["name"]} ({body["user"]["id"]})')
        logger.log("BODY:")
        logger.info(list(body.keys()))
        logger.log("OPTIONS:")
        logger.info(json.dumps(OPTIONS, indent=2))
        logger.log("MDOEL ID:")
        logger.info(model_id)
        logger.log("MDOEL:")
        logger.info(MODEL)
        logger.log("URL:")
        logger.info(URL)
        logger.log("MESSAGES:", len(messages), colors=["LOG", "INFO"])
        logger.log("TOKENS:", token_counter(
            messages, "mistral"), colors=["LOG", "INFO"])
        print("----- END INFO -----")

        try:
            r = requests.post(
                url=URL,
                json={**body, "model": MODEL, "options": OPTIONS},
                stream=True,
            )
            r.raise_for_status()

            if body.get("stream"):
                response_chunks = []

                def line_generator():
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")

                            # Load the line as JSON
                            try:
                                decoded_chunk = json.loads(decoded_line)
                                content = decoded_chunk["message"]["content"]
                                response_chunks.append(content)
                                logger.success(content, flush=True)

                                is_done = decoded_chunk["done"]
                                if is_done:
                                    logger.log("Last chunk")
                                    logger.debug(decoded_chunk)

                                yield content

                            except json.JSONDecodeError:
                                logger.warning(
                                    f"\nLast decoded line: {decoded_line}")

                    # Combine all content parts to form the final response
                    # combined_response = ''.join(response_chunks)
                    logger.log("RESPONSE CHUNKS:", len(
                        response_chunks), colors=["LOG", "SUCCESS"])

                return line_generator()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
