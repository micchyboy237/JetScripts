import asyncio
from jet.transformers.formatters import format_json
from IPython import display
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain import hub
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import chain as chain_decorator
from langgraph.graph import END, START, StateGraph
from playwright.async_api import Page
from playwright.async_api import async_playwright
from typing import List, Optional
from typing_extensions import TypedDict
import asyncio
import base64
import os
import platform
import re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Web Voyager

[WebVoyager](https://arxiv.org/abs/2401.13919) by He, et. al., is a vision-enabled web-browsing agent capable of controlling the mouse and keyboard.

It works by viewing annotated browser screenshots for each turn, then choosing the next step to take. The agent architecture is a basic reasoning and action (ReAct) loop. 
The unique aspects of this agent are:
- It's usage of [Set-of-Marks](https://som-gpt4v.github.io/)-like image annotations to serve as UI affordances for the agent
- It's application in the browser by using tools to control both the mouse and keyboard

The overall design looks like the following:

<img src="./img/web-voyager.excalidraw.jpg" src="../img/web-voyager.excalidraw.jpg" >

## Setup

First, let's install our required packages:
"""
logger.info("# Web Voyager")

# %%capture --no-stderr
# %pip install -U --quiet langgraph langsmith jet.llm.ollama.base_langchain

# from getpass import getpass


# def _getpass(env_var: str):
    if not os.environ.get(env_var):
#         os.environ[env_var] = getpass(f"{env_var}=")


# _getpass("OPENAI_API_KEY")

"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

#### Install Agent requirements

The only additional requirement we have is the [playwright](https://playwright.dev/) browser. Uncomment and install below:
"""
logger.info("#### Install Agent requirements")

# %pip install --upgrade --quiet  playwright > /dev/null
# !playwright install

# import nest_asyncio

# nest_asyncio.apply()

"""
## Helper File

We will use some JS code for this tutorial, which you should place in a file called `mark_page.js` in the same directory as the notebook you are running this tutorial from.

<div>
  <button type="button" style="border: 1px solid black; border-radius: 5px; padding: 5px; background-color: lightgrey;" onclick="toggleVisibility('helper-functions')">Show/Hide JS Code</button>
  <div id="helper-functions" style="display:none;">
    <!-- Helper functions -->
    <pre>

    const customCSS = `
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #27272a;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 0.375rem;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    `;

    const styleTag = document.createElement("style");
    styleTag.textContent = customCSS;
    document.head.append(styleTag);

    let labels = [];

    function unmarkPage() {
    // Unmark page logic
    for (const label of labels) {
        document.body.removeChild(label);
    }
    labels = [];
    }

    function markPage() {
    unmarkPage();

    var bodyRect = document.body.getBoundingClientRect();

    var items = Array.prototype.slice
        .call(document.querySelectorAll("*"))
        .map(function (element) {
        var vw = Math.max(
            document.documentElement.clientWidth || 0,
            window.innerWidth || 0
        );
        var vh = Math.max(
            document.documentElement.clientHeight || 0,
            window.innerHeight || 0
        );
        var textualContent = element.textContent.trim().replace(/\s{2,}/g, " ");
        var elementType = element.tagName.toLowerCase();
        var ariaLabel = element.getAttribute("aria-label") || "";

        var rects = [...element.getClientRects()]
            .filter((bb) => {
            var center_x = bb.left + bb.width / 2;
            var center_y = bb.top + bb.height / 2;
            var elAtCenter = document.elementFromPoint(center_x, center_y);

            return elAtCenter === element || element.contains(elAtCenter);
            })
            .map((bb) => {
            const rect = {
                left: Math.max(0, bb.left),
                top: Math.max(0, bb.top),
                right: Math.min(vw, bb.right),
                bottom: Math.min(vh, bb.bottom),
            };
            return {
                ...rect,
                width: rect.right - rect.left,
                height: rect.bottom - rect.top,
            };
            });

        var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);

        return {
            element: element,
            include:
            element.tagName === "INPUT" ||
            element.tagName === "TEXTAREA" ||
            element.tagName === "SELECT" ||
            element.tagName === "BUTTON" ||
            element.tagName === "A" ||
            element.onclick != null ||
            window.getComputedStyle(element).cursor == "pointer" ||
            element.tagName === "IFRAME" ||
            element.tagName === "VIDEO",
            area,
            rects,
            text: textualContent,
            type: elementType,
            ariaLabel: ariaLabel,
        };
        })
        .filter((item) => item.include && item.area >= 20);

    // Only keep inner clickable items
    items = items.filter(
        (x) => !items.some((y) => x.element.contains(y.element) && !(x == y))
    );

    // Function to generate random colors
    function getRandomColor() {
        var letters = "0123456789ABCDEF";
        var color = "#";
        for (var i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

    // Lets create a floating border on top of these elements that will always be visible
    items.forEach(function (item, index) {
        item.rects.forEach((bbox) => {
        newElement = document.createElement("div");
        var borderColor = getRandomColor();
        newElement.style.outline = `2px dashed ${borderColor}`;
        newElement.style.position = "fixed";
        newElement.style.left = bbox.left + "px";
        newElement.style.top = bbox.top + "px";
        newElement.style.width = bbox.width + "px";
        newElement.style.height = bbox.height + "px";
        newElement.style.pointerEvents = "none";
        newElement.style.boxSizing = "border-box";
        newElement.style.zIndex = 2147483647;
        // newElement.style.background = `${borderColor}80`;

        // Add floating label at the corner
        var label = document.createElement("span");
        label.textContent = index;
        label.style.position = "absolute";
        // These we can tweak if we want
        label.style.top = "-19px";
        label.style.left = "0px";
        label.style.background = borderColor;
        // label.style.background = "black";
        label.style.color = "white";
        label.style.padding = "2px 4px";
        label.style.fontSize = "12px";
        label.style.borderRadius = "2px";
        newElement.appendChild(label);

        document.body.appendChild(newElement);
        labels.push(newElement);
        // item.element.setAttribute("-ai-label", label.textContent);
        });
    });
    const coordinates = items.flatMap((item) =>
        item.rects.map(({ left, top, width, height }) => ({
        x: (left + left + width) / 2,
        y: (top + top + height) / 2,
        type: item.type,
        text: item.text,
        ariaLabel: item.ariaLabel,
        }))
    );
    return coordinates;
    }


</pre>
  </div>
</div>

<script>
  function toggleVisibility(id) {
    var element = document.getElementById(id);
    element.style.display = (element.style.display === "none") ? "block" : "none";
  }
</script>

## Define graph

### Define graph state

The state provides the inputs to each node in the graph.

In our case, the agent will track the webpage object (within the browser), annotated images + bounding boxes, the user's initial request, and the messages containing the agent scratchpad, system prompt, and other information.
"""
logger.info("## Helper File")




class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]


class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool

"""
### Define tools

The agent has 6 simple tools:

1. Click (at labeled box)
2. Type
3. Scroll
4. Wait
5. Go back
6. Go to search engine (Google)


We define them below here as functions:
"""
logger.info("### Define tools")



async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    async def run_async_code_42220dbe():
        await page.mouse.click(x, y)
        return 
     = asyncio.run(run_async_code_42220dbe())
    logger.success(format_json())
    return f"Clicked {bbox_id}"


async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return (
            f"Failed to type in element from bounding box labeled as number {type_args}"
        )
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    async def run_async_code_42220dbe():
        await page.mouse.click(x, y)
        return 
     = asyncio.run(run_async_code_42220dbe())
    logger.success(format_json())
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    async def run_async_code_0ca844e7():
        await page.keyboard.press(select_all)
        return 
     = asyncio.run(run_async_code_0ca844e7())
    logger.success(format_json())
    async def run_async_code_02be7f01():
        await page.keyboard.press("Backspace")
        return 
     = asyncio.run(run_async_code_02be7f01())
    logger.success(format_json())
    async def run_async_code_89ec148a():
        await page.keyboard.type(text_content)
        return 
     = asyncio.run(run_async_code_89ec148a())
    logger.success(format_json())
    async def run_async_code_b52c223f():
        await page.keyboard.press("Enter")
        return 
     = asyncio.run(run_async_code_b52c223f())
    logger.success(format_json())
    return f"Typed {text_content} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        async def run_async_code_1051d831():
            await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
            return 
         = asyncio.run(run_async_code_1051d831())
        logger.success(format_json())
    else:
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        async def run_async_code_ba71530b():
            await page.mouse.move(x, y)
            return 
         = asyncio.run(run_async_code_ba71530b())
        logger.success(format_json())
        async def run_async_code_28ea9c32():
            await page.mouse.wheel(0, scroll_direction)
            return 
         = asyncio.run(run_async_code_28ea9c32())
        logger.success(format_json())

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"


async def wait(state: AgentState):
    sleep_time = 5
    async def run_async_code_52fcb303():
        await asyncio.sleep(sleep_time)
        return 
     = asyncio.run(run_async_code_52fcb303())
    logger.success(format_json())
    return f"Waited for {sleep_time}s."


async def go_back(state: AgentState):
    page = state["page"]
    async def run_async_code_dd0561b3():
        await page.go_back()
        return 
     = asyncio.run(run_async_code_dd0561b3())
    logger.success(format_json())
    return f"Navigated back a page to {page.url}."


async def to_google(state: AgentState):
    page = state["page"]
    async def run_async_code_c3d63428():
        await page.goto("https://www.google.com/")
        return 
     = asyncio.run(run_async_code_c3d63428())
    logger.success(format_json())
    return "Navigated to google.com."

"""
### Define Agent

The agent is driven by a multi-modal model and decides the action to take for each step. It is composed of a few runnable objects:

1. A `mark_page` function to annotate the current page with bounding boxes
2. A prompt to hold the user question, annotated image, and agent scratchpad
3. GPT-4V to decide the next steps
4. Parsing logic to extract the action


Let's first define the annotation step:
#### Browser Annotations

This function annotates all buttons, inputs, text areas, etc. with numbered bounding boxes. GPT-4V then just has to refer to a bounding box
when taking actions, reducing the complexity of the overall task.
"""
logger.info("### Define Agent")



with open("mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page):
    async def run_async_code_8de24abf():
        await page.evaluate(mark_page_script)
        return 
     = asyncio.run(run_async_code_8de24abf())
    logger.success(format_json())
    for _ in range(10):
        try:
            async def run_async_code_107d2932():
                async def run_async_code_bf1c5f1f():
                    bboxes = await page.evaluate("markPage()")
                    return bboxes
                bboxes = asyncio.run(run_async_code_bf1c5f1f())
                logger.success(format_json(bboxes))
                return bboxes
            bboxes = asyncio.run(run_async_code_107d2932())
            logger.success(format_json(bboxes))
            break
        except Exception:
            asyncio.sleep(3)
    async def run_async_code_eab3a3ac():
        async def run_async_code_c6528812():
            screenshot = await page.screenshot()
            return screenshot
        screenshot = asyncio.run(run_async_code_c6528812())
        logger.success(format_json(screenshot))
        return screenshot
    screenshot = asyncio.run(run_async_code_eab3a3ac())
    logger.success(format_json(screenshot))
    async def run_async_code_b6ef54bc():
        await page.evaluate("unmarkPage()")
        return 
     = asyncio.run(run_async_code_b6ef54bc())
    logger.success(format_json())
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

"""
#### Agent definition

Now we'll compose this function with the prompt, llm and output parser to complete our agent.
"""
logger.info("#### Agent definition")



async def annotate(state):
    async def run_async_code_f7e9f398():
        async def run_async_code_d5839256():
            marked_page = await mark_page.with_retry().ainvoke(state["page"])
            return marked_page
        marked_page = asyncio.run(run_async_code_d5839256())
        logger.success(format_json(marked_page))
        return marked_page
    marked_page = asyncio.run(run_async_code_f7e9f398())
    logger.success(format_json(marked_page))
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}


prompt = hub.pull("wfh/web-voyager")

llm = ChatOllama(model="llama3.2")
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

"""
## Compile the graph

We've created most of the important logic. We have one more function to define that will help us update the graph state after a tool is called.
"""
logger.info("## Compile the graph")



def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}

"""
Now we can compose everything into a graph:
"""
logger.info("Now we can compose everything into a graph:")



graph_builder = StateGraph(AgentState)


graph_builder.add_node("agent", agent)
graph_builder.add_edge(START, "agent")

graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}


for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    graph_builder.add_edge(node_name, "update_scratchpad")


def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action


graph_builder.add_conditional_edges("agent", select_tool)

graph = graph_builder.compile()

"""
## Use the graph

Now that we've created the whole agent executor, we can run it on a few questions! We'll start our browser at "google.com" and then let it control the rest.

Below is a helper function to help print out the steps to the notebook (and display the intermediate screenshots).
"""
logger.info("## Use the graph")


async def run_async_code_9a3343bf():
    async def run_async_code_1cbd15a1():
        browser = await async_playwright().start()
        return browser
    browser = asyncio.run(run_async_code_1cbd15a1())
    logger.success(format_json(browser))
    return browser
browser = asyncio.run(run_async_code_9a3343bf())
logger.success(format_json(browser))
async def run_async_code_9e29a123():
    async def run_async_code_5e9400e8():
        browser = await browser.chromium.launch(headless=False, args=None)
        return browser
    browser = asyncio.run(run_async_code_5e9400e8())
    logger.success(format_json(browser))
    return browser
browser = asyncio.run(run_async_code_9e29a123())
logger.success(format_json(browser))
async def run_async_code_540e6be7():
    async def run_async_code_5b98ebc6():
        page = await browser.new_page()
        return page
    page = asyncio.run(run_async_code_5b98ebc6())
    logger.success(format_json(page))
    return page
page = asyncio.run(run_async_code_540e6be7())
logger.success(format_json(page))
async def run_async_code_0455a776():
    async def run_async_code_f877d721():
        _ = await page.goto("https://www.google.com")
        return _
    _ = asyncio.run(run_async_code_f877d721())
    logger.success(format_json(_))
    return _
_ = asyncio.run(run_async_code_0455a776())
logger.success(format_json(_))


async def call_agent(question: str, page, max_steps: int = 150):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    async for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        display.clear_output(wait=False)
        steps.append(f"{len(steps) + 1}. {action}: {action_input}")
        logger.debug("\n".join(steps))
        display.display(display.Image(base64.b64decode(event["agent"]["img"])))
        if "ANSWER" in action:
            final_answer = action_input[0]
            break
    return final_answer

async def run_async_code_6e90817f():
    async def run_async_code_679673cd():
        res = await call_agent("Could you explain the WebVoyager paper (on arxiv)?", page)
        return res
    res = asyncio.run(run_async_code_679673cd())
    logger.success(format_json(res))
    return res
res = asyncio.run(run_async_code_6e90817f())
logger.success(format_json(res))
logger.debug(f"Final response: {res}")

async def async_func_40():
    res = await call_agent(
        "Please explain the today's XKCD comic for me. Why is it funny?", page
    )
    return res
res = asyncio.run(async_func_40())
logger.success(format_json(res))
logger.debug(f"Final response: {res}")

async def run_async_code_efc618fe():
    async def run_async_code_ce1513bd():
        res = await call_agent("What are the latest blog posts from langchain?", page)
        return res
    res = asyncio.run(run_async_code_ce1513bd())
    logger.success(format_json(res))
    return res
res = asyncio.run(run_async_code_efc618fe())
logger.success(format_json(res))
logger.debug(f"Final response: {res}")

async def async_func_48():
    res = await call_agent(
        "Could you check google maps to see when i should leave to get to SFO by 7 o'clock? starting from SF downtown.",
        page,
    )
    return res
res = asyncio.run(async_func_48())
logger.success(format_json(res))
logger.debug(f"Final response: {res}")

logger.info("\n\n[DONE]", bright=True)