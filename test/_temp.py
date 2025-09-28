# jet_python_modules/jet/libs/haystack_integrations/llama_cpp/usage_example.py
from typing import Any, Dict, List, Optional, Union
from haystack import component
import requests
import logging
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

@component
class LlamaCppRemoteGenerator:
    """
    A generator that sends prompts to a remote llama.cpp server for text generation.
    """
    def __init__(
        self,
        server_url: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the remote generator.

        :param server_url: The URL of the llama.cpp server (e.g., 'http://shawn-pc.local:8080').
        :param generation_kwargs: Optional dictionary of generation parameters (e.g., max_tokens, temperature).
        """
        self.server_url = server_url.rstrip('/')
        self.generation_kwargs = generation_kwargs or {}
        self.session = requests.Session()

    def warm_up(self):
        """
        Test connection to the server.
        """
        try:
            response = self.session.get(f"{self.server_url}/health")
            response.raise_for_status()
            logger.info("Connected to llama.cpp server at %s", self.server_url)
        except requests.RequestException as e:
            logger.error("Failed to connect to llama.cpp server: %s", e)
            raise RuntimeError(f"Cannot connect to server at {self.server_url}: {e}")

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[List[str], List[Dict[str, Any]]]]:
        """
        Send the prompt to the remote llama.cpp server and retrieve the generated text.

        :param prompt: The input prompt for text generation.
        :param generation_kwargs: Optional override for generation parameters.
        :returns: A dictionary with 'replies' (list of generated texts) and 'meta' (list of response metadata).
        """
        if not prompt:
            return {"replies": [], "meta": []}

        # Combine default and provided generation kwargs
        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        payload = {
            "prompt": prompt,
            **updated_generation_kwargs
        }

        try:
            response = self.session.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            output = response.json()

            if not isinstance(output, dict) or "choices" not in output:
                raise ValueError(f"Unexpected response format from server: {output}")

            replies = [choice["text"] for choice in output["choices"]]
            meta = [dict(output.items())]
            return {"replies": replies, "meta": meta}

        except requests.RequestException as e:
            logger.error("Failed to generate text: %s", e)
            raise RuntimeError(f"Error communicating with llama.cpp server: {e}")

    def __del__(self):
        """
        Clean up the HTTP session.
        """
        self.session.close()

def run_text_example(prompt: str):
    """
    Run a text-only example using LlamaCppRemoteGenerator.
    """
    # Configure logging for better visibility
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Initialize the remote generator
    generator = LlamaCppRemoteGenerator(
        server_url="http://shawn-pc.local:8080",
        generation_kwargs={
            "n_ctx": 4096,
            "n_batch": 128,
            "max_tokens": 128,
            "temperature": 0.1,
        }
    )

    try:
        # Warm up the generator to ensure server connectivity
        generator.warm_up()

        # Define a sample prompt
        logger.info("Sending prompt: %s", prompt)

        # Run the generator with the prompt
        result = generator.run(
            prompt=prompt,
            generation_kwargs={"max_tokens": 50}  # Override max_tokens for this run
        )

        # Print the results
        replies = result["replies"]
        meta = result["meta"]
        logger.info("Generated replies: %s", replies)
        logger.debug("Response metadata: %s", meta)

        if replies:
            print(f"Answer: {replies[0]}")
        else:
            print("No reply received from the server.")

    except RuntimeError as e:
        logger.error("Error during generation: %s", e)
        print(f"Failed to generate response: {e}")

def run_image_example(image_path: str):
    """
    Run an example using LlamaCppRemoteGenerator with an image and text prompt.

    :param image_path: Path to the image file to send to the server.
    """
    # Configure logging for better visibility
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Initialize the remote generator
    generator = LlamaCppRemoteGenerator(
        server_url="http://shawn-pc.local:8080",
        generation_kwargs={
            "n_ctx": 4096,
            "n_batch": 128,
            "max_tokens": 128,
            "temperature": 0.1,
        }
    )

    try:
        # Warm up the generator to ensure server connectivity
        generator.warm_up()

        # Read and encode the image as base64
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Define a multimodal prompt
        prompt = "Describe the content of the provided image."
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    ]
                }
            ],
            **generator.generation_kwargs,
            "max_tokens": 100  # Override for this example
        }

        logger.info("Sending multimodal prompt with image: %s", image_path)

        # Send request to the chat completions endpoint
        response = generator.session.post(
            f"{generator.server_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        output = response.json()

        if not isinstance(output, dict) or "choices" not in output:
            raise ValueError(f"Unexpected response format from server: {output}")

        replies = [choice["message"]["content"] for choice in output["choices"]]
        meta = [dict(output.items())]
        logger.info("Generated replies: %s", replies)
        logger.debug("Response metadata: %s", meta)

        if replies:
            print(f"Answer: {replies[0]}")
        else:
            print("No reply received from the server.")

    except (FileNotFoundError, requests.RequestException, ValueError) as e:
        logger.error("Error during image-based generation: %s", e)
        print(f"Failed to generate response: {e}")

if __name__ == "__main__":
    # Run the text-only example
    print("Running text example...")
    prompt = "What is the capital of France?"
    run_text_example(prompt)

    # Run the image example with a sample image
    print("\nRunning image example...")
    sample_image_path = "sample_image.jpg"  # Replace with an actual image path
    run_image_example(sample_image_path)
