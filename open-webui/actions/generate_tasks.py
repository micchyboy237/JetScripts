import asyncio
import json
import requests
from typing import Callable, Optional, Union, Generator
from pydantic import BaseModel, Field
from jet.logger import logger


class Action:
    class Valves(BaseModel):
        openai_api_url: str = Field(
            default="http://localhost:11434",
            description="OpenAI-compatible endpoint",
        )
        model: str = Field(
            default="llama3.1:latest",
            description="Model to use for task generation",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    def extract_json(self, response: str) -> list:
        """Extract JSON blocks from the response."""
        import re
        matches = re.findall(
            r'```(?:json)?\s*\n(.*?)\n\s*```', response, re.DOTALL
        )

        if not matches:
            return ["No JSON found in response."]

        for match in matches:
            json_str = match.strip()
            try:
                json_obj = json.loads(json_str)
                return json_obj
            except json.JSONDecodeError:
                pass

    def query_openai_api(self, body: dict, stream: bool = True) -> Union[dict, Generator]:
        """Query the OpenAI API."""
        URL = f"{self.valves.openai_api_url}/api/chat"
        MODEL = self.valves.model
        OPTIONS = {
            "stream": stream,
            "num_ctx": 4096,
            "temperature": 0.7,
        }

        try:
            r = requests.post(
                url=URL,
                json={**body, "model": MODEL, "options": OPTIONS},
                stream=stream,
            )
            r.raise_for_status()

            if stream:
                response_chunks = []

                def line_generator():
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
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

                return line_generator()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

    async def action(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__: Callable = None,
    ) -> Optional[dict]:
        """Generate tasks for the provided epics."""
        epics = self.extract_json(body["messages"][-1]["content"])
        results = []

        for epic in epics:
            if isinstance(epic, dict):
                system_message = (
                    "You are responsible for breaking down epics into actionable tasks. "
                    "Each task must be specific, include clear acceptance criteria, and define dependencies if any. "
                    "Format response in JSON array wrapped in ```json."
                )
                user_prompt = (
                    f"Write tasks for this epic.\nTitle: {epic['title']}\nDescription: {epic['description']}")
                payload = {
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": True,
                }

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Generating tasks for epic: {epic['title']}", "done": False},
                        }
                    )

                stream_response = self.query_openai_api(payload, stream=True)
                tasks = []

                if isinstance(stream_response, Generator):
                    for content in stream_response:
                        try:
                            task_data = self.extract_json(content)
                            if task_data:
                                tasks.extend(task_data)
                        except Exception as e:
                            print(f"Error processing chunk: {e}")
                else:
                    print("Non-streaming response received.")

                results.append(
                    {"title": epic["title"], "description": epic["description"], "tasks": tasks})

        return {"results": results}


def main():
    async def run_action():
        action_instance = Action()
        body = {
            "messages": [
                {
                    "content": """
                    ```json
                    [
                        {
                            "title": "Build a User Authentication System",
                            "description": "Develop a secure and scalable authentication system for the web application."
                        },
                        {
                            "title": "Create a Payment Integration",
                            "description": "Integrate multiple payment gateways to support user transactions."
                        }
                    ]
                    ```
                    """
                }
            ]
        }

        async def event_emitter(event):
            print(f"Event Emitted: {event}")

        results = await action_instance.action(
            body=body,
            __user__={"valves": {"show_status": True}},
            __event_emitter__=event_emitter,
        )

        print("Final Results:", results)

    asyncio.run(run_action())


if __name__ == "__main__":
    main()
