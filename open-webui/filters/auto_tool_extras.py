"""
title: Auto-Tool v2
author: Wes Caldwell
email: Musicheardworldwide@gmail.com
date: 2024-07-19
version: 1.0
license: MIT
description: Auto-Tool function with extras.
"""

from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, List, Dict
import json

from open_webui.apps.webui.models.users import Users
from open_webui.apps.webui.models.tools import Tools
from open_webui.apps.webui.models.models import Models

from open_webui.main import generate_chat_completions
from open_webui.utils.misc import get_last_user_message


class Filter:
    class Valves(BaseModel):
        template: str = Field(
            default="""Tools: {{TOOLS}}
If a tool doesn't match the query, return an empty list []. Otherwise, return a list of matching tool IDs in the format ["tool_id"]. Select multiple tools if applicable. Only return the list. Do not return any other text. Review the entire chat history to ensure the selected tool matches the context. If unsure, default to an empty list []. Use tools conservatively."""
        )
        status: bool = Field(default=False)
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.user_history = {}
        self.tool_analytics = {}
        self.user_feedback = {}
        self.models = {}
        pass

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        messages = body["messages"]
        user_message = get_last_user_message(messages)

        if self.valves.status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Finding the right tools...",
                        "done": False,
                    },
                }
            )

        all_tools = [
            {"id": tool.id, "description": tool.meta.description}
            for tool in Tools.get_tools()
        ]
        available_tool_ids = (
            __model__.get("info", {}).get("meta", {}).get("toolIds", [])
        )
        available_tools = [
            tool for tool in all_tools if tool["id"] in available_tool_ids
        ]

        # Tool Recommendation Based on User History
        user_id = __user__["id"]
        if user_id in self.user_history:
            recommended_tools = [
                tool
                for tool in available_tools
                if tool["id"] in self.user_history[user_id]
            ]
        else:
            recommended_tools = available_tools

        # Dynamic Tool Filtering
        filtered_tools = self.filter_tools(recommended_tools, user_message)

        system_prompt = self.valves.template.replace(
            "{{TOOLS}}", str(filtered_tools))
        prompt = (
            "History:\n"
            + "\n".join(
                [
                    f"{message['role'].upper()}: \"\"\"{
                        message['content']}\"\"\""
                    for message in messages[::-1][:4]
                ]
            )
            + f"\nQuery: {user_message}"
        )

        payload = {
            "model": body["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        try:
            user = Users.get_user_by_id(user_id)
            response = await generate_chat_completions(form_data=payload, user=user)
            content = response["choices"][0]["message"]["content"]

            # Parse the function response
            if content is not None:
                print(f"content: {content}")
                content = content.replace("'", '"')
                result = json.loads(content)

                if isinstance(result, list) and len(result) > 0:
                    body["tool_ids"] = result
                    if self.valves.status:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Found matching tools: {', '.join(result)}",
                                    "done": True,
                                },
                            }
                        )
                    # Update Tool Usage Analytics
                    self.update_tool_analytics(result)
                else:
                    if self.valves.status:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "No matching tools found.",
                                    "done": True,
                                },
                            }
                        )

        except Exception as e:
            print(e)
            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error processing request: {e}",
                            "done": True,
                        },
                    }
                )
            pass

        return body

    def filter_tools(self, tools: List[Dict], query: str) -> List[Dict]:
        # Implement dynamic tool filtering based on user-defined criteria
        filtered_tools = []
        for tool in tools:
            if query.lower() in tool["description"].lower():
                filtered_tools.append(tool)
        return filtered_tools

    def update_tool_analytics(self, tool_ids: List[str]) -> None:
        for tool_id in tool_ids:
            if tool_id in self.tool_analytics:
                self.tool_analytics[tool_id] += 1
            else:
                self.tool_analytics[tool_id] = 1

    def integrate_user_feedback(self, user_id: str, feedback: Dict) -> None:
        if user_id in self.user_feedback:
            self.user_feedback[user_id].append(feedback)
        else:
            self.user_feedback[user_id] = [feedback]

    def add_model(self, model_id: str, model_info: Dict) -> None:
        self.models[model_id] = model_info

    def get_model(self, model_id: str) -> Optional[Dict]:
        return self.models.get(model_id)
