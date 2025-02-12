import os
import requests
import time
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")

    def __init__(self):
        self.type = "manifold"
        self.id = "ollama"
        self.name = "ollama/"
        self.valves = self.Valves()
        self.default_model = "llama3"

    def get_ollama_models(self):
        return [{"id": self.default_model, "name": self.default_model}]

    def pipes(self) -> List[dict]:
        return self.get_ollama_models()

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []

        for message in messages:
            processed_content = ""
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content = item["text"]
            else:
                processed_content = message.get("content", "")

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        payload = {
            "model": body.get("model", self.default_model),
            "messages": processed_messages,
            "stream": body.get("stream", False),
        }

        url = f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions"

        try:
            if body.get("stream", False):
                return self.stream_response(url, payload)
            else:
                return self.non_stream_response(url, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, payload):
        try:
            response = requests.post(
                url=url,
                json=payload,
                stream=True,
            )

            if response.status_code != 200:
                raise Exception(
                    f"HTTP Error {response.status_code}: {response.text}")

            # Process the streamed response
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    yield chunk.decode("utf-8")
                time.sleep(0.01)  # Throttle streaming to avoid overloading

        except requests.exceptions.RequestException as e:
            print(f"Stream request failed: {e}")
            yield f"Error: {e}"
        except Exception as e:
            print(f"Error in stream_response method: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, payload):
        try:
            response = requests.post(
                url=url, json=payload, timeout=(3.05, 60)
            )
            if response.status_code != 200:
                raise Exception(
                    f"HTTP Error: {response.status_code}: {response.text}"
                )

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
        except Exception as e:
            print(f"Error in non_stream_response method: {e}")
            return f"Error: {e}"
