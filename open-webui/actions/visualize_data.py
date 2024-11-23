"""
title: Make charts out of your data v2
author: Iqbal Maulana
author_url: https://github.com/iqballx?tab=repositories
author_linkedin: https://www.linkedin.com/in/iqbaalm/
funding_url: https://github.com/open-webui
version: 2.0.0
"""

import asyncio
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Generator
import os
from open_webui.apps.webui.models.files import Files
import uuid
import logging
import time
import json
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BUILD_CHARTS = """

Objective:
Your goal is to read the query, extract the data, choose the appropriate chart to present the data, and produce the HTML to display it.

Steps:

	1.	Read and Examine the Query:
	•	Understand the user’s question and identify the data provided.
	2.	Analyze the Data:
	•	Examine the data in the query to determine the appropriate chart type (e.g., bar chart, pie chart, line chart) for effective visualization.
	3.	Generate HTML:
	•	Create the HTML code to present the data using the selected chart format.
	4.	Handle No Data Situations:
	•	If there is no data in the query or the data cannot be presented as a chart, generate a humorous or funny HTML response indicating that the data cannot be presented.
    5.	Calibrate the chart scale based on the data:
	•	based on the data try to make the scale of the chart as readable as possible.

Key Considerations:

	-	Your output should only include HTML code, without any additional text.
    -   Generate only HTML. Do not include any additional words or explanations.
    -   Make to remove any character other non alpha numeric from the data.
    -   is the generated HTML Calibrate the chart scale based on the data for eveything to be readable.
    -   Generate only html code , nothing else , only html.


Example1 : 
'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Chart</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="chart" style="width: 100%; height: 100vh;"></div>
    <button id="save-button">Save Screenshot</button>
    <script>
        // Data for the chart
        var data = [{
            x: [''Category 1'', ''Category 2'', ''Category 3''],
            y: [20, 14, 23],
            type: ''bar''
        }];

        // Layout for the chart
        var layout = {
            title: ''Interactive Bar Chart'',
            xaxis: {
                title: ''Categories''
            },
            yaxis: {
                title: ''Values''
            }
        };

        // Render the chart
        Plotly.newPlot(''chart'', data, layout);

        // Function to save screenshot
        document.getElementById(''save-button'').onclick = function() {
            Plotly.downloadImage(''chart'', {format: ''png'', width: 800, height: 600, filename: ''chart_screenshot''});
        };

        // Function to update chart attributes
        function updateChartAttributes(newData, newLayout) {
            Plotly.react(''chart'', newData, newLayout);
        }

        // Example of updating chart attributes
        var newData = [{
            x: [''New Category 1'', ''New Category 2'', ''New Category 3''],
            y: [10, 22, 30],
            type: ''bar''
        }];

        var newLayout = {
            title: ''Updated Bar Chart'',
            xaxis: {
                title: ''New Categories''
            },
            yaxis: {
                title: ''New Values''
            }
        };

        // Call updateChartAttributes with new data and layout
        // updateChartAttributes(newData, newLayout);
    </script>
</body>
</html>
'''

Example2:
'''
<!DOCTYPE html>
<html>
<head>
    <title>Employees by Job/Function</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="myChart" style="width: 100%; max-width: 700px; height: 500px; margin: 0 auto;"></div>
    <script>
        var data = [{
            x: ["System Engineer", "Solution Analyst", "Development Engineer", "Squad Leader", "Enterprise Architect", "Tech Lead", "Technical Architect", "Methods/Tools Expert"],
            y: [5, 3, 2, 1, 1, 1, 1, 1],
            type: "bar",
            marker: {
                color: "rgb(49,130,189)"
            }
        }];
        var layout = {
            title: "STT Employees by Job/Function",
            xaxis: {
                title: "Job/Function"
            },
            yaxis: {
                title: "Number of Employees"
            }
        };
        Plotly.newPlot("myChart", data, layout);
    </script>
</body>
</html>
'''

2.	No Data or Unchartable Data:
''' 
<html>
<body>
    <h1>We're sorry, but your data can't be charted.</h1>
    <p>Maybe try feeding it some coffee first?</p>
    <img src="https://media.giphy.com/media/l4EoTHjkw0XiYtNRG/giphy.gif" alt="Funny Coffee GIF">
</body>
</html>

'''

"""
USER_PROMPT_GENERATE_HTML = """
Giving this query  {Query} generate the necessary html qurty.
"""


def query_openai_api(
    model: str,
    system_prompt: str,
    prompt: str,
    openai_api_url: str,
) -> Generator[str, None, None]:
    """Queries the OpenAI-compatible API and streams the response using requests."""
    url = f"{openai_api_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        with requests.post(url, headers=headers, json=payload, stream=payload['stream']) as response:
            if payload['stream']:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        data_str = decoded_line[len("data: "):]

                        # Load the line as JSON
                        try:
                            decoded_json = json.loads(data_str)
                            choice = decoded_json.get("choices", [{}])[0]
                            content = choice.get(
                                "delta", {}).get("content", "")
                            yield content
                        except json.JSONDecodeError:
                            print(f"Last decoded line: {data_str}")
            else:
                return response.json()
    except Exception as e:
        logger.error(f"Error: {e}")
        yield f"Error: {e}"


class FileData(BaseModel):
    id: str
    filename: str
    meta: Dict[str, Any]
    path: str


class Action:
    class Valves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        html_filename: str = Field(
            default="json_visualizer.html",
            description="Name of the HTML file to be created or retrieved.",
        )
        OPENIA_KEY: str = Field(
            default="",
            description="key to consume OpenIA interface like LLM for example a litellm key.",
        )
        OPENIA_URL: str = Field(
            default="http://jetairm1:11434",
            description="Host where to consume the OpenAI interface like llm",
        )
        MODEL_NAME: str = Field(
            default="qwen2.5-coder:latest",
            description="model name",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.openai = None
        self.html_content = """

        """

    def create_or_get_file(self, user_id: str, json_data: str) -> str:

        filename = str(int(time.time() * 1000)) + self.valves.html_filename
        directory = "action_embed"

        logger.debug(f"Attempting to create or get file: {filename}")

        # Check if the file already exists
        existing_files = Files.get_files()
        for file in existing_files:
            if (
                file.filename == f"{directory}/{user_id}/{filename}"
                and file.user_id == user_id
            ):
                logger.debug(f"Existing file found. Updating content.")
                # Update the existing file with new JSON data
                self.update_html_content(file.meta["path"], json_data)
                return file.id

        # If the file doesn''t exist, create it
        generated_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/open-webui/generated"
        base_path = os.path.join(generated_dir, directory)
        os.makedirs(base_path, exist_ok=True)
        file_path = os.path.join(base_path, filename)

        logger.debug(f"Creating new file at: {file_path}")
        self.update_html_content(file_path, json_data)

        file_id = str(uuid.uuid4())
        meta = {
            "source": file_path,
            "title": "Modern JSON Visualizer",
            "content_type": "text/html",
            "size": os.path.getsize(file_path),
            "path": file_path,
        }

        # Create a new file entry
        file_data = FileData(
            id=file_id,
            filename=f"{directory}/{user_id}/{filename}",
            meta=meta,
            path=file_path,
        )
        new_file = Files.insert_new_file(user_id, file_data)
        logger.debug(f"New file created with ID: {new_file.id}")
        return new_file.id

    def update_html_content(self, file_path: str, html_content: str):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.debug(f"HTML content updated at: {file_path}")

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        logger.debug(f"action:{__name__} started")
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Analyzing Data",
                    "done": False,
                },
            }
        )
        try:
            original_content = body["messages"][-1]["content"]
            system_prompt = SYSTEM_PROMPT_BUILD_CHARTS
            user_prompt = USER_PROMPT_GENERATE_HTML.format(
                Query=original_content)

            # Query OpenAI-compatible API
            html_content = ""
            results = query_openai_api(
                model=self.valves.MODEL_NAME,
                system_prompt=system_prompt,
                prompt=user_prompt,
                openai_api_url=self.valves.OPENIA_URL,
            )
            for result in results:
                print(result, end="", flush=True)
                html_content += result

            # Create or get file
            user_id = __user__["id"]
            file_id = self.create_or_get_file(user_id, html_content)
            html_embed_tag = f"{{{{HTML_FILE_ID_{file_id}}}}}"
            body["messages"][-1]["content"] = (
                f"{original_content}\n\n{html_embed_tag}")

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Visualizing the chart",
                        "done": True,
                    },
                }
            )
            logger.debug("Objects visualized")
        except Exception as e:
            error_message = f"Error visualizing JSON: {str(e)}"
            logger.error(f"Error: {error_message}")
            body["messages"][-1]["content"] += f"\n\nError: {error_message}"
            if self.valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error Visualizing JSON",
                            "done": True,
                        },
                    }
                )
        logger.debug(f"action:{__name__} completed")
        return body


def main():
    # Sample query with real-world data
    sample_query = {
        "title": "Sales Data",
        "data": {
            "categories": ["Q1", "Q2", "Q3", "Q4"],
            "values": [15000, 20000, 25000, 30000],
        }
    }

    # Example user information
    user_info = {
        "id": "test_user",
        "name": "Test User"
    }

    # Action class instance
    action_instance = Action()

    async def execute_test():
        # Prepare a fake body structure for simulation
        body = {
            "messages": [
                {
                    "content": f"Please visualize the following data: {sample_query}"
                }
            ]
        }

        # Mock event emitter
        async def event_emitter(event):
            print(f"Event: {event}")

        # Call the action method
        response = await action_instance.action(
            body=body,
            __user__=user_info,
            __event_emitter__=event_emitter
        )

        # Print the resulting HTML or error
        print("\n--- Generated Output ---\n")
        print(response["messages"][-1]["content"])

    # Run the async test
    asyncio.run(execute_test())


if __name__ == "__main__":
    main()
