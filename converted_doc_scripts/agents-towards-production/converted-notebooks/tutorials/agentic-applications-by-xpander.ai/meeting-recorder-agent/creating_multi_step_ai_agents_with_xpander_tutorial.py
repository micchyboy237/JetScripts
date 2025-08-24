from datetime import datetime
from dotenv import load_dotenv
from jet.logger import CustomLogger
from openai import MLX
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from xpander_sdk import Agent, XpanderClient, LLMProvider, ToolCallResult, Tokens, LLMTokens
import os
import shutil
import tempfile
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--agentic-applications-by-xpander-ai--meeting-recorder-agent--creating-multi-step-ai-agents-with-xpander-tutorial)

# Creating Multi-Step AI Agents Using xpander.ai

### Overview

This tutorial walks you through building a complex, multi-step AI agent using a streamlined SDK and visual UI, without managing backend infrastructure. Using a real-world example, you’ll see how easy it is to connect tools, manage state, and deploy production-ready agents with minimal setup.

### What is xpander.ai?

xpander.ai is a Backend-as-a-Service platform for deploying and managing autonomous AI agents, without the overhead of backend infrastructure. It offers built-in memory, tool orchestration, and visual development tools, allowing you to move from prototype to production seamlessly.

### Key Features & Benefits

- 🧠 **Persistent Memory & State** – Built-in storage for long-term agent context  
- 🔧 **Tool Orchestration** – Easily connect prebuilt tools or any API  
- 🧪 **Visual Workbench** – Test, trace, and wire tools and prompts without code  
- 🚀 **Multi-Channel Triggers** – Run agents from Slack, Webhooks, UIs, and more  
- 🔍 **Observability & Tracing** – Inspect tool calls, model decisions, and payloads  
- 🔒 **Secure Model Access** – Bring your own keys or use managed models  
- ☁️ **Cloud-Native Runtime** – Scale agents reliably in production  
- 🔄 **Composable Components** – Reuse tools and logic across agents  
- ⚙️ **Zero Infrastructure Setup** – No need to manage queues, DBs, or servers

In short, xpander.ai helps you build robust, real-world AI agents quickly and efficiently.

### What You’ll Build

This tutorial demonstrates a multi-agent setup built using xpander.ai. The example focuses on automating a meeting recording workflow, but it's just one possible use case. The framework is flexible, and you're free to build agents tailored to your own tools, data, and business logic.

In this example, you’ll walk through creating a multi-step AI agent that:

1. Connects to your Google Calendar to find upcoming meetings  
2. Schedules and initiates meeting recordings  
3. Checks recorder status and retrieves post-meeting assets (video & transcript)  
4. Emails summaries and recordings to you or your team  
5. Maintains memory and context across sessions  
6. Uses simple Python functions as tools the agent can call directly  

This example demonstrates how xpander.ai’s SDK and visual Workbench simplify the process of building, testing, and deploying production-grade agents. You'll combine low code and full code approaches with built-in observability and a fully managed runtime.

Let’s dive in and see xpander.ai in action!

### Installation

Before coding, make sure you have the following:

- **Python** ≥ 3.10 installed on your system

First, let's install the required packages:
"""
logger.info("# Creating Multi-Step AI Agents Using xpander.ai")

# %pip install -r requirements.txt

"""
# Set Up and Build the AI Agent

## 1. Register and Log In to xpander.ai

To get started, visit [https://app.xpander.ai](https://app.xpander.ai/login?utm=atp) and sign up for a **Free Account**.

If you already have an account, simply log in with your credentials.

<div style="text-align: center;">
<img src="assets/images/xpander-login.png" width="800">
</div>

<br>

> 💡 **Tip:** xpander.ai supports login with Google or Github accounts.

---

## 2. Preview: What You'll See After Importing the Agent

Once you import the agent template (coming up in the next step), your screen will look something like this:

<div style="text-align: center;">
<img src="assets/images/agent-screenshot.png" width="900">
</div>

<div style="
  border-left: 4px solid #42b983;
  padding-left: 1em;
  margin: 1em 0;
  font-size: 14px;
">
  <strong>💡 </strong> Quick rundown of the xpander.ai Agent Workbench:

  * The top nodes are source nodes - you use them to configure different ways to trigger your agent.

  * The box marked "Meeting Recording Agent - (Imported)" is a visual representation of your agent.

  * Nodes inside the agent are tools (for function calling). Every time the agent is triggered, these tools are presented to the model as its available tools in the inference call.
  
  * You can click the Settings (⚙️) button to see all the settings for your agent, such as instructions, memory settings, conversation starters, and more.
</div>

----

## 3. Import the Prebuilt Agent Template

We’ve created a preconfigured template that includes everything you need to run the Meeting Recorder Agent.

### ⚠️ Important:

- You will be redirected to the xpander.ai platform.
- Follow the steps to import the template into your workspace.
- Name the agent (or keep the default).
- Once it's loaded, **return to this tutorial** 

We’ll guide you through using it step-by-step.

<p align="center">
  <a href="https://app.xpander.ai/templates/48039a71-c99c-4691-8b66-a6faca3ccbe4?utm=atp" target="_blank">
    <img src="assets/images/agent-import-button.svg" 
         alt="Import Meeting Recorder Agent Template" 
         width="320" />
  </a>
</p>

 💡 After completing the template import, make sure your agent looks similar to the image above, and that you complete the connection to Google Calendar (make sure there's no red exclamation mark (❗) on your Google Calendar tool)

## 4. Using the Agent from the UI

Once you’ve imported the Meeting Recorder Agent, it’s ready to use no extra setup required.

In addition to running it from code, you can test and interact with your agent directly from the xpander.ai platform using a simple user interface.

This is perfect for quick testing or for non-technical users who want to try out the agent’s capabilities.

---

### ▶️ Demo 1: Testing the Agent via the Workbench

This demo shows how to use the **Test Agent** tab in the xpander.ai Workbench.

You'll see how to start a conversation with the agent, ask it to schedule a meeting recorder, and watch how it selects and runs the right tool.

📍 The Test Agent tab is perfect for debugging and testing tool-based workflows.

<video controls width="100%">
  <source src="assets/videos/meeting-recorder.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

### ▶️ Demo 2: Triggering the Agent from the Chat UI

This demo shows how to interact with your agent through the **Chat UI**, just like you would with a chatbot or assistant.

📍 This interface is great for real world usage or sharing your agent with others.

<video controls width="100%">
  <source src="assets/videos/post-meeting.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 5. Create Your .env File

To run your agent from code, create a .env file in your project root with these values:

* XPANDER_API_KEY=your_xpander_api_key 
* XPANDER_AGENT_ID=your_imported_agent_id
# * OPENAI_API_KEY=your_openai_api_key

- **Get XPANDER keys**: Open your imported agent on xpander.ai platform, click the **SDK** node (top bar), and copy both keys.  
- **Get OPENAI key**: From [MLX’s API keys page](https://platform.openai.com/account/api-keys)

## 6. Add Your Own Python Function as a Local Tool

In this step, you'll learn how to connect a custom Python function to your agent and make it callable  just like built-in tools.

This means you can now extend your agent with any Python logic you want, without deploying APIs or writing backend infrastructure.

### 6.1 Required Imports and Setup

First, let's import all the necessary libraries and set up our environment.
"""
logger.info("# Set Up and Build the AI Agent")



load_dotenv()
xpander_client = XpanderClient(api_key=os.environ['XPANDER_API_KEY'])
# openai_client = MLX(api_key=os.environ['OPENAI_API_KEY'])

"""
### 6.2 Creating the PDF Generation Local Function

Now, let's create our custom function that generates a PDF meeting schedule. This function takes a list of meetings and creates a well-formatted PDF agenda with the following features:

- Chronological sorting of meetings
- Clean table layout with headers
- Formatted date and time display
- Participant list formatting
- Automatic saving to Downloads folder
"""
logger.info("### 6.2 Creating the PDF Generation Local Function")

def export_meeting_schedule_pdf(meetings: list) -> str:
    """
    Generate a clean, well-formatted PDF agenda for weekly meetings and save it to the user's Downloads folder.
    Returns the full path to the saved PDF.

    Args:
        meetings (list): List of meeting dictionaries containing:
            - title (str): Meeting title
            - start_time (str): ISO 8601 formatted start time
            - end_time (str): ISO 8601 formatted end time
            - location (str, optional): Meeting location
            - participants (list, optional): List of participant names/emails
    """
    if not meetings:
        return "No meetings provided."

    meetings.sort(key=lambda m: m.get("start_time", ""))

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_pdf.name, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    title = Paragraph("<b>Weekly Meeting Agenda</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    cell_style = ParagraphStyle(name='Cell', fontSize=10, leading=12, alignment=TA_LEFT)

    data = [["Date", "Time", "Title", "Location", "Participants"]]

    for meeting in meetings:
        try:
            start_dt = datetime.fromisoformat(meeting["start_time"])
            end_dt = datetime.fromisoformat(meeting["end_time"])
            date = start_dt.strftime("%A, %b %d")
            time = f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}"
        except Exception:
            date = time = "Invalid"

        title = Paragraph(meeting.get("title", "Untitled"), cell_style)
        location = Paragraph(meeting.get("location", "—"), cell_style)
        attendees = meeting.get("participants") or meeting.get("attendees") or []
        participants = Paragraph(", ".join(attendees) if isinstance(attendees, list) else str(attendees), cell_style)

        data.append([date, time, title, location, participants])

    table = Table(data, colWidths=[3.5*cm, 3*cm, 5*cm, 4*cm, 5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#6741d9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('VALIGN', (0, 1), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 1), (-1, -1), 6),
        ('RIGHTPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])
    ]))

    elements.append(table)
    doc.build(elements)

    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads_path, exist_ok=True)

    filename = f"weekly_meeting_agenda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    final_path = os.path.join(downloads_path, filename)
    shutil.copy(temp_pdf.name, final_path)

    return final_path

"""
### 6.3 Defining the Tool Schema

Now, let's define the tool schema that enables LLM function calling. This schema acts as a contract between your Python function and the LLM, telling the model:

- How to understand and use your function
- What parameters it needs to provide
- What data types to expect
- How to format the function call

This schema follows the MLX function calling format, allowing the LLM to properly call your Python function when needed.
"""
logger.info("### 6.3 Defining the Tool Schema")

local_tools = [{
    "declaration": {
        "type": "function",
        "function": {
            "name": "export_meeting_schedule_pdf",
            "description": "Generate a weekly meeting agenda as a PDF from a list of meetings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meetings": {
                        "type": "array",
                        "description": "List of meetings to include in the agenda.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Title of the meeting"
                                },
                                "start_time": {
                                    "type": "string",
                                    "description": "Start time in ISO 8601 format"
                                },
                                "end_time": {
                                    "type": "string",
                                    "description": "End time in ISO 8601 format"
                                },
                                "location": {
                                    "type": "string",
                                    "description": "Meeting location (optional)"
                                },
                                "participants": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of participant names or emails"
                                }
                            },
                            "required": ["title", "start_time", "end_time"]
                        }
                    }
                },
                "required": ["meetings"]
            }
        }
    },
    "fn": export_meeting_schedule_pdf
}]

local_tools_list = [tool['declaration'] for tool in local_tools]
local_tools_by_name = {tool['declaration']['function']['name']: tool['fn']
                       for tool in local_tools}

"""
## 7. Running the Agent

In this step, you'll learn how to run your agent with both built-in and custom tools. The code is split into three main functions that work together:

### 7.1 Setting Up the Agent

First, let's create a function to load your agent from xpander.ai:
"""
logger.info("## 7. Running the Agent")

def setup_agent() -> Agent:
    """load the meeting recording Agent from xpander.ai"""
    agent = xpander_client.agents.get(agent_id=os.environ['XPANDER_AGENT_ID'])
    logger.debug(f"🔄 Loaded agent: {agent.name}")
    logger.debug(f"🔍 View this agent in the Xpander platform with ID: https://app.xpander.ai/agents/{agent.id}")
    return agent

"""
### 7.2 The Agent Loop

The core loop that drives your agent’s behavior. It:

- Sends the agent’s messages to the LLM (`qwen3-1.7b-4bit`)
- Tracks token usage and reports it to xpander.ai
- Handles tool calls:
  - ☁️ Built-in/cloud tools
  - 🛠️ Your custom local tools (like the PDF generator)
- Feeds tool results back into memory for multi-step reasoning
- Ends when the agent’s task is complete
"""
logger.info("### 7.2 The Agent Loop")

def agent_loop(agent: Agent, local_tools_by_name=None):
    logger.debug("🪄 Starting Agent Loop")
    execution_tokens = Tokens(worker=LLMTokens(completion_tokens=0, prompt_tokens=0, total_tokens=0))
    execution_start_time = time.perf_counter()

    while not agent.is_finished():
        start_time = time.perf_counter()
        response = openai_client.chat.completions.create(
            model="qwen3-1.7b-4bit",
            messages=agent.messages,
            tools=agent.get_tools(llm_provider=LLMProvider.OPEN_AI),
            tool_choice=agent.tool_choice,
            temperature=0
        )

        execution_tokens.worker.completion_tokens += response.usage.completion_tokens
        execution_tokens.worker.prompt_tokens += response.usage.prompt_tokens
        execution_tokens.worker.total_tokens += response.usage.total_tokens

        agent.report_llm_usage(
            llm_response=response.model_dump(),
            llm_inference_duration=time.perf_counter() - start_time,
            llm_provider=LLMProvider.OPEN_AI
        )

        agent.add_messages(response.model_dump())

        tool_calls = XpanderClient.extract_tool_calls(
            llm_response=response.model_dump(),
            llm_provider=LLMProvider.OPEN_AI
        )

        if tool_calls:
            for call in tool_calls:
                name = getattr(call, 'name', None) or getattr(getattr(call, 'function', {}), 'name', "unnamed")
                logger.debug(f"🔧 Using tool: {name}")

            agent.run_tools(tool_calls=tool_calls)

            if local_tools_by_name:
                pending_local_tool_execution = XpanderClient.retrieve_pending_local_tool_calls(tool_calls=tool_calls)
                local_tools_results = []

                for tc in pending_local_tool_execution:
                    logger.debug(f"🛠️ Running local tool: {tc.name}")
                    tool_call_result = ToolCallResult(function_name=tc.name, tool_call_id=tc.tool_call_id, payload=tc.payload)
                    try:
                        if tc.name in local_tools_by_name:
                            tool_call_result.is_success = True
                            tool_call_result.result = local_tools_by_name[tc.name](**tc.payload)
                        else:
                            raise Exception(f"Local tool {tc.name} not found")
                    except Exception as e:
                        tool_call_result.is_success = False
                        tool_call_result.is_error = True
                        tool_call_result.result = str(e)
                    finally:
                        local_tools_results.append(tool_call_result)

                if local_tools_results:
                    logger.debug(f"📝 Registering {len(local_tools_results)} local tool results...")
                    agent.memory.add_tool_call_results(tool_call_results=local_tools_results)

    agent.report_execution_metrics(
        llm_tokens=execution_tokens,
        ai_model="qwen3-1.7b-4bit"
    )

    logger.debug(f"✨ Execution duration: {time.perf_counter() - execution_start_time:.2f} seconds")
    logger.debug(f"🔢 Total tokens used: {execution_tokens.worker.total_tokens}")

"""
#### Understanding the Agent Loop

To help visualize the complex logic inside the `agent_loop` function, here's a flowchart showing how it processes tasks and makes decisions:

<div style="text-align: center;">
<img src="assets/images/agent_loop_visualization.png" width=1000">
</div>
<br>

#### Key Components Explained

1. **Initialization** - Sets up tracking and timing for the agent's execution

2. **Main Loop** - Manages the conversation flow and tool execution until completion

3. **LLM Interaction** - Handles communication with MLX and tracks usage metrics

4. **Tool Management** - Processes both cloud and local tools, updating agent memory

5. **Metrics & Reporting** - Tracks and reports execution metrics to xpander.ai

#### Flow Control

- 🔄 **Loop Control**: The agent continues processing until `is_finished()` returns true
- 🔍 **Tool Detection**: Automatically identifies when tools need to be called
- 🛠️ **Tool Execution**: Handles both cloud-based and local tools
- 📊 **Metrics**: Tracks and reports all important metrics throughout execution

This visualization helps understand how the agent processes tasks, interacts with tools, and maintains state throughout its execution cycle.

### 7.3 Chat Interface

Finally, let's create a simple chat interface to interact with your agent.

The `chat()` function provides a simple interface to:
- Send messages to your agent
- Run the full agent loop
- Get responses and maintain conversation context
- Use both built-in and custom tools
"""
logger.info("#### Understanding the Agent Loop")

def chat(agent: Agent, message, thread_id=None, local_tools_by_name=None):
    """Send a message to the agent and get a response"""
    logger.debug(f"\n👤 User: {message}")

    agent.add_task(input=message, thread_id=thread_id)

    agent_loop(agent, local_tools_by_name)

    result = agent.retrieve_execution_result()
    logger.debug(f"🤖 Agent: {result.result}")
    logger.debug(f"🧵 Thread ID: {result.memory_thread_id}")
    return result.memory_thread_id

"""
## 8. Usage Examples: Interacting with the Agent

Now that the agent is set up and ready, let’s try a few example prompts to see how it behaves.

We'll start by asking the agent a simple question to understand its capabilities:

**"Hi! What can you do?"**

This will:

- Start a new memory thread
- Send the prompt to the agent
- Print the response based on the agent’s instructions

More examples will follow to demonstrate how the agent uses tools, manages context, and completes real tasks.
"""
logger.info("## 8. Usage Examples: Interacting with the Agent")

agent = setup_agent()
agent.add_local_tools(local_tools_list)
logger.debug("🧰 Local tools added to agent")
thread_id = chat(agent, 'Hi! What can you do?', local_tools_by_name=local_tools_by_name)

"""
##### Add Calendar Integration:
Now let’s try a real use case: asking the agent to retrieve your upcoming meetings from Google Calendar.

This example reuses the same thread created earlier (so the agent maintains context).  
You simply send a follow-up prompt asking for meeting details on a specific date.

---

<div style="border-left: 4px solid #42b983; padding-left: 1em; margin: 1em 0; font-size: 14px; line-height: 1.6">
  <p>
    <strong>🚨 Replace</strong> <code>&lt;DATE&gt;</code> <strong>with the actual date you want to query.</strong><br>
    <span> Format examples:</span> <code>May 21 2025</code>, <code>2025-05-21</code><br>
    <span> Example prompt:</span><br>
    <code>List my upcoming meetings on May 21 2025 and the three consecutive days...</code>
  </p>
</div>
"""
logger.info("##### Add Calendar Integration:")

chat(agent, 'List my upcoming meetings on <DATE> and the three consecutive days, for each meeting, include: title, description (if available), location, time, participants', thread_id, local_tools_by_name=local_tools_by_name)

"""
##### Run Our Local Tool – Meeting Schedule PDF Calendar Example

Now let’s use the local tool we just added — `export_meeting_schedule_pdf` — to create a PDF that summarizes your upcoming meetings for the week in a clean, formatted table.

This example sends a follow-up prompt using the same memory thread, and triggers your custom local Python function from within the agent.

When you run the code below, the agent will:

- Continue the conversation from the current thread
- Use your registered local tool to process meeting data
- Generate and return a downloadable PDF agenda
"""
logger.info("##### Run Our Local Tool – Meeting Schedule PDF Calendar Example")

chat(agent, "Create meeting schedule for the upcoming 3 days, and export it as a PDF", thread_id, local_tools_by_name=local_tools_by_name)

"""
##### Recording Control Integration

In this step, you’ll ask the agent to create a meeting recorder for one of your scheduled meetings and then check the status to retrieve the recording assets.

This example continues using the same memory thread so the agent can keep track of your previous calendar request and match the correct meeting.

---

<div style="border-left: 4px solid #42b983; padding-left: 1em; margin: 1em 0; font-size: 14px; line-height: 1.8">
  <p>
    <strong>🚨 Replace</strong> <code>&lt;MEETING_TITLE&gt;</code> <strong>with the exact title of the meeting you want to create a recorder for, from the list above.</strong><br>
    <span> Example prompt:</span><br>
    <code>Create a recorder for the Q2 Planning Sync.</code>
  </p>
</div>

---

When you run the code below, the agent will:

- Attempt to create a recorder for the meeting title you specify
- Track the status of the recorder
- Return asset links (e.g. transcript or video) once the recording is complete
"""
logger.info("##### Recording Control Integration")

chat(agent, 'Create a recorder for the <MEETING_TITLE>.', thread_id, local_tools_by_name=local_tools_by_name)

chat(agent, 'Check the recorder status and give me the asset links if done.', thread_id, local_tools_by_name=local_tools_by_name)

"""
#### Email Delivery

In this step, you'll ask the agent to email the recording assets (video and transcript) to a specific address along with a summary.

Since the agent maintains context from earlier steps, it knows which meeting was recorded and where the assets came from.

---

<div style="border-left: 4px solid #42b983; padding-left: 1em; margin: 1em 0; font-size: 14px; line-height: 1.6">
  <p>
    <strong>🚨 Replace</strong> <code>&lt;YOUR_EMAIL&gt;</code> <strong>with your actual email address.</strong><br>
    <span> Example prompt:</span><br>
    <code>Email the video & transcript to team@example.com with a summary.</code>
  </p>
</div>
"""
logger.info("#### Email Delivery")

chat(agent, 'Email the video & transcript to <YOUR_EMAIL> with a summary.', thread_id, local_tools_by_name=local_tools_by_name)

"""
## Next Steps - Building on Top of This Agent:

This agent is just a starting point. You can extend it with xpander.ai by:

- testing in the Visual Workbench 
- adding tools or changing the existing tools to other calendars
- sending outputs via Slack, email, or webhooks 
- switching LLMs without code changes 
- managing state automatically 
- triggering from API, A2A, Slack, or Web UI 
- monitoring everything with built-in observability and version control. 

📚 See the [xpander docs](https://docs.xpander.ai?utm=atp) for deeper integrations.

## Conclusion

In this tutorial, you’ve built a fully functional AI Meeting Recorder powered by xpander.ai and MLX. Along the way, we:

1. Initialized a multi-step agent using xpander.ai
2. Integrated Google Calendar for meeting discovery
3. Triggered meeting recording and retrieved assets
4. Summarized and emailed meeting notes all through an intelligent agent

xpander.ai provides a powerful platform for building AI agents by simplifying service integrations and providing enterprise-ready features. This approach makes it easy to create, deploy, and scale AI-powered meeting solutions.
"""
logger.info("## Next Steps - Building on Top of This Agent:")

logger.info("\n\n[DONE]", bright=True)