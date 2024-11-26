from typing import Any, Awaitable, Callable, Optional
from pydantic import BaseModel, Field
import aiohttp
import json


def extract_json(response):
    import re
    matches = re.findall(r'```(?:json)\s*\n(.*?)\n\s*```', response, re.DOTALL)

    if not matches:
        return "No json found in response."

    results = []
    for match in matches:
        json_str = match.strip()
        try:
            json_obj = json.loads(json_str)
            results.append(json_obj)
        except:
            results.append("Invalid json.")

    return results


class Filter:
    class Valves(BaseModel):
        openai_api_url: str = Field(
            default="http://jetairm1:11434",
            description="openai compatible endpoint",
        )
        model: str = Field(
            default="llama3.1:latest",
            description="Model to use for task generation",
        )
        pass

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        print(f"inlet:{__name__}")
        print(f"inlet:body:{body}")
        print(f"inlet:user:{__user__}")
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        print(f"outlet:{__name__}")
        print(f"outlet:body:{body}")
        print(f"outlet:user:{__user__}")

        last_message = body["messages"][-1]["content"]
        epics = extract_json(last_message)

        for epic in epics:
            epic_title = epic.get("title")
            epic_description = epic.get("description")

            if epic_title and epic_description:
                tasks = await self.generate_tasks(epic_title, epic_description)
                if tasks.startswith("[") and tasks.endswith("]") and len(tasks) != 2:
                    result = await self.process_tasks(tasks)

                    if __user__["valves"].show_status:
                        status_description = (
                            f"Generated tasks successfully: {tasks}"
                            if result
                            else f"Task generation failed: {result}"
                        )
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": status_description,
                                    "done": True,
                                },
                            }
                        )
        return body

    async def generate_tasks(self, epic_title: str, epic_description: str) -> str:
        system_prompt = "1. You are responsible for breaking down epics into actionable tasks. 2. Each task must be specific, include clear acceptance criteria, and define dependencies if any. 3. Tasks must be scoped appropriately to allow completion within a reasonable timeframe.\nFormat your response as a valid JSON format surrounded by ```json."
        user_message = (
            f"Write tasks for this epic.\nTitle: {epic_title}\nDescription: {epic_description}")
        return await self.query_openai_api(self.valves.model, system_prompt, user_message)

    async def query_openai_api(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
    ) -> str:
        url = f"{self.valves.openai_api_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        print(f"query_openai_api:url:{url}")
        print(f"query_openai_api:headers:\n{json.dumps(headers, indent=2)}")
        print(f"query_openai_api:payload:\n{json.dumps(payload, indent=2)}")
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(url, headers=headers, json=payload)
                response.raise_for_status()
                json_content = await response.json()
                content = json_content["choices"][0]["message"]["content"]
                print(f"tasks:\n{json.dumps(content, indent=2)}")
            return content
        except aiohttp.ClientError as e:
            raise Exception(f"Http error: {e.response.text}")

    async def process_tasks(self, tasks: str) -> bool:
        """Given a list of tasks as a string, validate and store each task."""
        try:
            task_list = json.loads(tasks)
            for task in task_list:
                print(f"Processing task: {task}")
                # Store each task in your database or system here
            return True
        except Exception as e:
            return f"Error processing tasks: {e}"
