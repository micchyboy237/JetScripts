import asyncio
from jet.transformers.formatters import format_json
from autogen.io.websockets import IOWebsockets
from contextlib import asynccontextmanager  # noqa: E402
from datetime import datetime
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from http.server import HTTPServer, SimpleHTTPRequestHandler  # noqa: E402
from jet.logger import CustomLogger
from pathlib import Path  # noqa: E402
from tempfile import TemporaryDirectory
from websockets.sync.client import connect as ws_connect
import autogen
import os
import shutil
import uvicorn  # noqa: E402


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Websockets: Streaming input and output using websockets

This notebook demonstrates how to use the [`IOStream`](https://autogenhub.github.io/autogen/docs/reference/io/base/IOStream) class to stream both input and output using websockets. The use of websockets allows you to build web clients that are more responsive than the one using web methods. The main difference is that the webosockets allows you to push data while you need to poll the server for new response using web mothods.


In this guide, we explore the capabilities of the [`IOStream`](https://autogenhub.github.io/autogen/docs/reference/io/base/IOStream) class. It is specifically designed to enhance the development of clients such as web clients which use  websockets for streaming both input and output. The [`IOStream`](https://autogenhub.github.io/autogen/docs/reference/io/base/IOStream) stands out by enabling a more dynamic and interactive user experience for web applications.

Websockets technology is at the core of this functionality, offering a significant advancement over traditional web methods by allowing data to be "pushed" to the client in real-time. This is a departure from the conventional approach where clients must repeatedly "poll" the server to check for any new responses. By employing the underlining [websockets](https://websockets.readthedocs.io/) library, the IOStream class facilitates a continuous, two-way communication channel between the server and client. This ensures that updates are received instantly, without the need for constant polling, thereby making web clients more efficient and responsive.

The real power of websockets, leveraged through the [`IOStream`](https://autogenhub.github.io/autogen/docs/reference/io/base/IOStream) class, lies in its ability to create highly responsive web clients. This responsiveness is critical for applications requiring real-time data updates such as chat applications. By integrating the [`IOStream`](https://autogenhub.github.io/autogen/docs/reference/io/base/IOStream) class into your web application, you not only enhance user experience through immediate data transmission but also reduce the load on your server by eliminating unnecessary polling.

In essence, the transition to using websockets through the [`IOStream`](https://autogenhub.github.io/autogen/docs/reference/io/base/IOStream) class marks a significant enhancement in web client development. This approach not only streamlines the data exchange process between clients and servers but also opens up new possibilities for creating more interactive and engaging web applications. By following this guide, developers can harness the full potential of websockets and the [`IOStream`](https://autogenhub.github.io/autogen/docs/reference/io/base/IOStream) class to push the boundaries of what is possible with web client responsiveness and interactivity.

## Requirements

````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install autogen[websockets] fastapi uvicorn
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("# Websockets: Streaming input and output using websockets")




config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    },
)

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

## Defining `on_connect` function

An `on_connect` function is a crucial part of applications that utilize websockets, acting as an event handler that is called whenever a new client connection is established. This function is designed to initiate any necessary setup, communication protocols, or data exchange procedures specific to the newly connected client. Essentially, it lays the groundwork for the interactive session that follows, configuring how the server and the client will communicate and what initial actions are to be taken once a connection is made. Now, let's delve into the details of how to define this function, especially in the context of using the AutoGen framework with websockets.


Upon a client's connection to the websocket server, the server automatically initiates a new instance of the [`IOWebsockets`](https://autogenhub.github.io/autogen/docs/reference/io/websockets/IOWebsockets) class. This instance is crucial for managing the data flow between the server and the client. The `on_connect` function leverages this instance to set up the communication protocol, define interaction rules, and initiate any preliminary data exchanges or configurations required for the client-server interaction to proceed smoothly.
"""
logger.info("## Defining `on_connect` function")

def on_connect(iostream: IOWebsockets) -> None:
    logger.debug(f" - on_connect(): Connected to client using IOWebsockets {iostream}", flush=True)

    logger.debug(" - on_connect(): Receiving message from client.", flush=True)

    initial_msg = iostream.input()

    agent = autogen.ConversableAgent(
        name="chatbot",
        system_message="Complete a task given to you and reply TERMINATE when the task is done. If asked about the weather, use tool 'weather_forecast(city)' to get the weather forecast for a city.",
        llm_config={
            "config_list": autogen.config_list_from_json(
                env_or_file="OAI_CONFIG_LIST",
                filter_dict={
                    "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
                },
            ),
            "stream": True,
        },
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        system_message="A proxy for the user.",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )

    def weather_forecast(city: str) -> str:
        return f"The weather forecast for {city} at {datetime.now()} is sunny."

    autogen.register_function(
        weather_forecast, caller=agent, executor=user_proxy, description="Weather forecast for a city"
    )

    logger.debug(
        f" - on_connect(): Initiating chat with agent {agent} using message '{initial_msg}'",
        flush=True,
    )
    user_proxy.initiate_chat(  # noqa: F704
        agent,
        message=initial_msg,
    )

"""
Here's an explanation on how a typical `on_connect` function such as the one in the example above is defined:

1. **Receive Initial Message**: Immediately after establishing a connection, receive an initial message from the client. This step is crucial for understanding the client's request or initiating the conversation flow.

2. **Instantiate ConversableAgent**: Create an instance of ConversableAgent with a specific system message and the LLM configuration. If you need more than one agent, make sure they don't share the same `llm_config` as 
adding a function to one of them will also attempt to add it to another.

2. **Instantiate UserProxyAgent**: Similarly, create a UserProxyAgent instance, defining its termination condition, human input mode, and other relevant parameters. There is no need to define `llm_config` as the UserProxyAgent
does not use LLM.

4. **Define Agent-specific Functions**: If your conversable agent requires executing specific tasks, such as fetching a weather forecast in the example above, define these functions within the on_connect scope. Decorate these functions accordingly to link them with your agents.

5. **Initiate Conversation**: Finally, use the `initiate_chat` method of your `UserProxyAgent` to start the interaction with the conversable agent, passing the initial message and a cache mechanism for efficiency.

## Testing websockets server with Python client

Testing an `on_connect` function with a Python client involves simulating a client-server interaction to ensure the setup, data exchange, and communication protocols function as intended. Here’s a brief explanation on how to conduct this test using a Python client:

1. **Start the Websocket Server**: Use the `IOWebsockets.run_server_in_thread method` to start the server in a separate thread, specifying the on_connect function and the port. This method returns the URI of the running websocket server.

2. **Connect to the Server**: Open a connection to the server using the returned URI. This simulates a client initiating a connection to your websocket server.

3. **Send a Message to the Server**: Once connected, send a message from the client to the server. This tests the server's ability to receive messages through the established websocket connection.

4. **Receive and Process Messages**: Implement a loop to continuously receive messages from the server. Decode the messages if necessary, and process them accordingly. This step verifies the server's ability to respond back to the client's request.

This test scenario effectively evaluates the interaction between a client and a server using the `on_connect` function, by simulating a realistic message exchange. It ensures that the server can handle incoming connections, process messages, and communicate responses back to the client, all critical functionalities for a robust websocket-based application.
"""
logger.info("## Testing websockets server with Python client")

with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8765) as uri:
    logger.debug(f" - test_setup() with websocket server running on {uri}.", flush=True)

    with ws_connect(uri) as websocket:
        logger.debug(f" - Connected to server on {uri}", flush=True)

        logger.debug(" - Sending message to server.", flush=True)
        websocket.send("Check out the weather in Paris and write a poem about it.")

        while True:
            message = websocket.recv()
            message = message.decode("utf-8") if isinstance(message, bytes) else message

            logger.debug(message, end="", flush=True)

            if "TERMINATE" in message:
                logger.debug()
                logger.debug(" - Received TERMINATE message. Exiting.", flush=True)
                break

"""
## Testing websockets server running inside FastAPI server with HTML/JS client

The code snippets below outlines an approach for testing an `on_connect` function in a web environment using [FastAPI](https://fastapi.tiangolo.com/) to serve a simple interactive HTML page. This method allows users to send messages through a web interface, which are then processed by the server running the AutoGen framework via websockets. Here's a step-by-step explanation:

1. **FastAPI Application Setup**: The code initiates by importing necessary libraries and setting up a FastAPI application. FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.

2. **HTML Template for User Interaction**: An HTML template is defined as a multi-line Python string, which includes a basic form for message input and a script for managing websocket communication. This template creates a user interface where messages can be sent to the server and responses are displayed dynamically.

3. **Running the Websocket Server**: The `run_websocket_server` async context manager starts the websocket server using `IOWebsockets.run_server_in_thread` with the specified `on_connect` function and port. This server listens for incoming websocket connections.

4. **FastAPI Route for Serving HTML Page**: A FastAPI route (`@app.get("/")`) is defined to serve the HTML page to users. When a user accesses the root URL, the HTML content for the websocket chat is returned, allowing them to interact with the websocket server.

5. **Starting the FastAPI Application**: Lastly, the FastAPI application is started using Uvicorn, an ASGI server, configured with the app and additional parameters as needed. The server is then launched to serve the FastAPI application, making the interactive HTML page accessible to users.

This method of testing allows for interactive communication between the user and the server, providing a practical way to demonstrate and evaluate the behavior of the on_connect function in real-time. Users can send messages through the webpage, and the server processes these messages as per the logic defined in the on_connect function, showcasing the capabilities and responsiveness of the AutoGen framework's websocket handling in a user-friendly manner.
"""
logger.info("## Testing websockets server running inside FastAPI server with HTML/JS client")



PORT = 8000

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Autogen websocket test</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8080/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@asynccontextmanager
async def run_websocket_server(app):
    with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8080) as uri:
        logger.debug(f"Websocket server started at {uri}.", flush=True)

        yield


app = FastAPI(lifespan=run_websocket_server)


@app.get("/")
async def get():
    return HTMLResponse(html)


config = uvicorn.Config(app)
server = uvicorn.Server(config)
async def run_async_code_aec30df9():
    await server.serve()  # noqa: F704
    return 
 = asyncio.run(run_async_code_aec30df9())
logger.success(format_json())

"""
The testing setup described above, leveraging FastAPI and websockets, not only serves as a robust testing framework for the on_connect function but also lays the groundwork for developing real-world applications. This approach exemplifies how web-based interactions can be made dynamic and real-time, a critical aspect of modern application development.

For instance, this setup can be directly applied or adapted to build interactive chat applications, real-time data dashboards, or live support systems. The integration of websockets enables the server to push updates to clients instantly, a key feature for applications that rely on the timely delivery of information. For example, a chat application built on this framework can support instantaneous messaging between users, enhancing user engagement and satisfaction.

Moreover, the simplicity and interactivity of the HTML page used for testing reflect how user interfaces can be designed to provide seamless experiences. Developers can expand upon this foundation to incorporate more sophisticated elements such as user authentication, message encryption, and custom user interactions, further tailoring the application to meet specific use case requirements.

The flexibility of the FastAPI framework, combined with the real-time communication enabled by websockets, provides a powerful toolset for developers looking to build scalable, efficient, and highly interactive web applications. Whether it's for creating collaborative platforms, streaming services, or interactive gaming experiences, this testing setup offers a glimpse into the potential applications that can be developed with these technologies.

## Testing  websockets server with HTML/JS client

The provided code snippet below is an example of how to create an interactive testing environment for an `on_connect` function using Python's built-in `http.server` module. This setup allows for real-time interaction within a web browser, enabling developers to test the websocket functionality in a more user-friendly and practical manner. Here's a breakdown of how this code operates and its potential applications:

1. **Serving a Simple HTML Page**: The code starts by defining an HTML page that includes a form for sending messages and a list to display incoming messages. JavaScript is used to handle the form submission and websocket communication.

2. **Temporary Directory for HTML File**: A temporary directory is created to store the HTML file. This approach ensures that the testing environment is clean and isolated, minimizing conflicts with existing files or configurations.

3. **Custom HTTP Request Handler**: A custom subclass of `SimpleHTTPRequestHandler` is defined to serve the HTML file. This handler overrides the do_GET method to redirect the root path (`/`) to the `chat.html` page, ensuring that visitors to the server's root URL are immediately presented with the chat interface.

4. **Starting the Websocket Server**: Concurrently, a websocket server is started on a different port using the `IOWebsockets.run_server_in_thread` method, with the previously defined `on_connect` function as the callback for new connections.

5. **HTTP Server for the HTML Interface**: An HTTP server is instantiated to serve the HTML chat interface, enabling users to interact with the websocket server through a web browser.

This setup showcases a practical application of integrating websockets with a simple HTTP server to create a dynamic and interactive web application. By using Python's standard library modules, it demonstrates a low-barrier entry to developing real-time applications such as chat systems, live notifications, or interactive dashboards.

The key takeaway from this code example is how easily Python's built-in libraries can be leveraged to prototype and test complex web functionalities. For developers looking to build real-world applications, this approach offers a straightforward method to validate and refine websocket communication logic before integrating it into larger frameworks or systems. The simplicity and accessibility of this testing setup make it an excellent starting point for developing a wide range of interactive web applications.
"""
logger.info("## Testing  websockets server with HTML/JS client")


PORT = 8000

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Autogen websocket test</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8080/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""

with TemporaryDirectory() as temp_dir:
    path = Path(temp_dir) / "chat.html"
    with open(path, "w") as f:
        f.write(html)

    class MyRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=temp_dir, **kwargs)

        def do_GET(self):
            if self.path == "/":
                self.path = "/chat.html"
            return SimpleHTTPRequestHandler.do_GET(self)

    handler = MyRequestHandler

    with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8080) as uri:
        logger.debug(f"Websocket server started at {uri}.", flush=True)

        with HTTPServer(("", PORT), handler) as httpd:
            logger.debug("HTTP server started at http://localhost:" + str(PORT))
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                logger.debug(" - HTTP server stopped.", flush=True)

logger.info("\n\n[DONE]", bright=True)