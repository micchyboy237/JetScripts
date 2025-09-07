async def main():
    from jet.transformers.formatters import format_json
    from a2a.client import ClientConfig, ClientFactory, create_text_message_object
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore, BasePushNotificationSender
    from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TransportProtocol,
    )
    from a2a.types import TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent
    from a2a.utils import new_agent_text_message, new_task, new_text_artifact
    from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
    from dotenv import load_dotenv
    from enum import Enum
    from jet.logger import CustomLogger
    from pydantic import BaseModel
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
    from semantic_kernel.connectors.ai.ollama import (
    OllamaChatCompletion,
    OllamaChatPromptExecutionSettings,
    )
    from semantic_kernel.contents import (
    FunctionCallContent,
    FunctionResultContent,
    StreamingTextContent,
    )
    from semantic_kernel.functions import KernelArguments, kernel_function
    from typing import Any, Annotated, AsyncIterable, Literal
    import asyncio
    import httpx
    import json
    import logging
    import os
    import shutil
    import threading
    import time
    import uvicorn
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Semantic Kernel with Agent-to-Agent (A2A) Protocol using Azure Ollama
    
    This notebook demonstrates how to use Semantic Kernel with the A2A protocol to create a multi-agent travel planning system using Azure Ollama via Azure AI Foundry. To setup your environment variables, you can follow the [Setup Lesson ](/00-course-setup/README.md)
    
    ## What You'll Build
    
    A three-agent travel planning system:
    1. **Currency Exchange Agent** - Handles currency conversion using real-time exchange rates
    2. **Activity Planner Agent** - Plans activities and provides travel recommendations
    3. **Travel Manager Agent** - Orchestrates the other agents to provide comprehensive travel assistance
    
    ## Installation
    
    First, let's install the required dependencies:
    
    ## Import Required Libraries
    """
    logger.info("# Semantic Kernel with Agent-to-Agent (A2A) Protocol using Azure Ollama")
    
    
    # import nest_asyncio
    
    
    
    """
    ## Environment Configuration
    
    Configure Azure Ollama settings. Make sure you have the following environment variables set:
    - `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`
    - `AZURE_OPENAI_ENDPOINT`
    # - `AZURE_OPENAI_API_KEY`
    """
    logger.info("## Environment Configuration")
    
    load_dotenv()
    
    # nest_asyncio.apply()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    
    logger.debug('Environment configured successfully!')
    
    """
    ## Define the Currency Plugin
    
    This plugin provides real-time currency exchange rates using the Frankfurter API.
    """
    logger.info("## Define the Currency Plugin")
    
    class CurrencyPlugin:
        """A currency plugin that leverages Frankfurter API for exchange rates."""
    
        @kernel_function(
            description='Retrieves exchange rate between currency_from and currency_to using Frankfurter API'
        )
        def get_exchange_rate(
            self,
            currency_from: Annotated[str, 'Currency code to convert from, e.g. USD'],
            currency_to: Annotated[str, 'Currency code to convert to, e.g. EUR or INR'],
            date: Annotated[str, "Date or 'latest'"] = 'latest',
        ) -> str:
            try:
                response = httpx.get(
                    f'https://api.frankfurter.app/{date}',
                    params={'from': currency_from, 'to': currency_to},
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()
                if 'rates' not in data or currency_to not in data['rates']:
                    return f'Could not retrieve rate for {currency_from} to {currency_to}'
                rate = data['rates'][currency_to]
                return f'1 {currency_from} = {rate} {currency_to}'
            except Exception as e:
                return f'Currency API call failed: {str(e)}'
    
    logger.debug('✅ Currency Plugin defined')
    
    """
    ## Define Response Format
    
    Structured response format for agent outputs.
    """
    logger.info("## Define Response Format")
    
    class ResponseFormat(BaseModel):
        """A Response Format model to direct how the model should respond."""
        status: Literal['input_required', 'completed', 'error'] = 'input_required'
        message: str
    
    logger.debug('✅ Response format defined')
    
    """
    ## Create the A2A Agent Executor
    
    This wraps Semantic Kernel agents to work with the A2A protocol.
    """
    logger.info("## Create the A2A Agent Executor")
    
    class SemanticKernelTravelAgentExecutor(AgentExecutor):
        """A2A Executor for Semantic Kernel Travel Agent."""
    
        def __init__(self):
            self.chat_service = OllamaChatCompletion(
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            )
    
            self.currency_agent = ChatCompletionAgent(
                service=self.chat_service,
                name='CurrencyExchangeAgent',
                instructions=(
                    'You specialize in handling currency-related requests from travelers. '
                    'This includes providing current exchange rates, converting amounts between different currencies, '
                    'explaining fees or charges related to currency exchange, and giving advice on the best practices for exchanging currency. '
                    'Your goal is to assist travelers promptly and accurately with all currency-related questions.'
                ),
                plugins=[CurrencyPlugin()],
            )
    
            self.activity_agent = ChatCompletionAgent(
                service=self.chat_service,
                name='ActivityPlannerAgent',
                instructions=(
                    'You specialize in planning and recommending activities for travelers. '
                    'This includes suggesting sightseeing options, local events, dining recommendations, '
                    'booking tickets for attractions, advising on travel itineraries, and ensuring activities '
                    'align with traveler preferences and schedule. '
                    'Your goal is to create enjoyable and personalized experiences for travelers.'
                ),
            )
    
            self.travel_agent = ChatCompletionAgent(
                service=self.chat_service,
                name='TravelManagerAgent',
                instructions=(
                    "Your role is to carefully analyze the traveler's request and forward it to the appropriate agent based on the "
                    'specific details of the query. '
                    'Forward any requests involving monetary amounts, currency exchange rates, currency conversions, fees related '
                    'to currency exchange, financial transactions, or payment methods to the CurrencyExchangeAgent. '
                    'Forward requests related to planning activities, sightseeing recommendations, dining suggestions, event '
                    'booking, itinerary creation, or any experiential aspects of travel that do not explicitly involve monetary '
                    'transactions to the ActivityPlannerAgent. '
                    'Your primary goal is precise and efficient delegation to ensure travelers receive accurate and specialized '
                    'assistance promptly.'
                ),
                plugins=[self.currency_agent, self.activity_agent],
            )
    
            self.thread = None
            self.SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
    
        async def execute(self, context, event_queue):
            """Execute method required by A2A framework."""
            try:
    
                user_input = context.get_user_input()
                task = context.current_task
    
                if not task:
                    task = new_task(context.message)
                    await event_queue.enqueue_event(task)
    
                session_id = task.context_id
                await self._ensure_thread_exists(session_id)
    
                response = await self.travel_agent.get_response(
                        messages=user_input,
                        thread=self.thread,
                    )
                logger.success(format_json(response))
    
                content = response.content if isinstance(response.content, str) else str(response.content)
    
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        append=False,
                        context_id=task.context_id,
                        task_id=task.id,
                        last_chunk=True,
                        artifact=new_text_artifact(
                            name='travel_result',
                            description='Travel planning result',
                            text=content,
                        ),
                    )
                )
    
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(state=TaskState.completed),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
    
            except Exception as e:
                logger.error(f"Error in SemanticKernelTravelAgentExecutor.execute: {str(e)}")
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=new_agent_text_message(
                                f"Error processing request: {str(e)}",
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
    
        async def cancel(self, context, event_queue):
            """Cancel method - not supported for this agent."""
            raise Exception('cancel not supported')
    
        async def _ensure_thread_exists(self, session_id: str) -> None:
            """Ensure thread exists for the session."""
            if self.thread is None or self.thread.id != session_id:
                if self.thread:
                    await self.thread.delete()
                self.thread = ChatHistoryAgentThread(thread_id=session_id)
    
    
    logger.debug('✅ Travel Manager Agent Executor simplified - removed JSON formatting constraints')
    
    """
    ## Create Individual A2A Agents
    
    Now we'll create A2A wrappers for each specialized agent.
    """
    logger.info("## Create Individual A2A Agents")
    
    class CurrencyAgentExecutor(AgentExecutor):
        """A2A Executor for Currency Exchange Agent."""
    
        def __init__(self):
            self.chat_service = OllamaChatCompletion(
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            )
    
            self.agent = ChatCompletionAgent(
                service=self.chat_service,
                name='CurrencyExchangeAgent',
                instructions=(
                    'You are a currency exchange specialist. Provide accurate exchange rates and currency conversion information. '
                    'Use the get_exchange_rate function to get real-time rates. '
                    'Always provide clear, concise information about currency conversions.'
                ),
                plugins=[CurrencyPlugin()],
            )
            self.thread = None
            self.SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
    
        async def execute(self, context, event_queue):
            """Execute method required by A2A framework."""
            try:
    
                user_input = context.get_user_input()
                task = context.current_task
    
                if not task:
                    task = new_task(context.message)
                    await event_queue.enqueue_event(task)
    
                session_id = task.context_id
                if self.thread is None or self.thread.id != session_id:
                    if self.thread:
                        await self.thread.delete()
                    self.thread = ChatHistoryAgentThread(thread_id=session_id)
    
                response = await self.agent.get_response(messages=user_input, thread=self.thread)
                logger.success(format_json(response))
                content = response.content if isinstance(response.content, str) else str(response.content)
    
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        append=False,
                        context_id=task.context_id,
                        task_id=task.id,
                        last_chunk=True,
                        artifact=new_text_artifact(
                            name='currency_result',
                            description='Currency exchange information',
                            text=content,
                        ),
                    )
                )
    
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(state=TaskState.completed),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
    
            except Exception as e:
                logger.error(f"Error in CurrencyAgentExecutor.execute: {str(e)}")
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=new_agent_text_message(
                                f"Error processing request: {str(e)}",
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
    
        async def cancel(self, context, event_queue):
            """Cancel method - not supported for this simple agent."""
            raise Exception('cancel not supported')
    
    class ActivityAgentExecutor(AgentExecutor):
        """A2A Executor for Activity Planner Agent."""
    
        def __init__(self):
            self.chat_service = OllamaChatCompletion(
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            )
    
            self.agent = ChatCompletionAgent(
                service=self.chat_service,
                name='ActivityPlannerAgent',
                instructions=(
                    'You are a travel activity planning specialist. Create detailed, personalized activity recommendations. '
                    'Include specific times, locations, and practical tips. '
                    'Consider budget, preferences, and local culture in your suggestions.'
                ),
            )
            self.thread = None
            self.SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
    
        async def execute(self, context, event_queue):
            """Execute method required by A2A framework."""
            try:
    
                user_input = context.get_user_input()
                task = context.current_task
    
                if not task:
                    task = new_task(context.message)
                    await event_queue.enqueue_event(task)
    
                session_id = task.context_id
                if self.thread is None or self.thread.id != session_id:
                    if self.thread:
                        await self.thread.delete()
                    self.thread = ChatHistoryAgentThread(thread_id=session_id)
    
                response = await self.agent.get_response(messages=user_input, thread=self.thread)
                logger.success(format_json(response))
                content = response.content if isinstance(response.content, str) else str(response.content)
    
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        append=False,
                        context_id=task.context_id,
                        task_id=task.id,
                        last_chunk=True,
                        artifact=new_text_artifact(
                            name='activity_result',
                            description='Activity planning recommendations',
                            text=content,
                        ),
                    )
                )
    
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(state=TaskState.completed),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
    
            except Exception as e:
                logger.error(f"Error in ActivityAgentExecutor.execute: {str(e)}")
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=new_agent_text_message(
                                f"Error processing request: {str(e)}",
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
    
        async def cancel(self, context, event_queue):
            """Cancel method - not supported for this simple agent."""
            raise Exception('cancel not supported')
    
    """
    ## Define Agent Cards
    
    Agent cards describe the capabilities of each agent for A2A discovery.
    """
    logger.info("## Define Agent Cards")
    
    currency_agent_card = AgentCard(
        name='Currency Exchange Agent',
        url='http://localhost:10020',
        description='Provides real-time currency exchange rates and conversion services',
        version='1.0',
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        preferred_transport=TransportProtocol.jsonrpc,
        skills=[
            AgentSkill(
                id='currency_exchange',
                name='Currency Exchange',
                description='Get exchange rates and convert between currencies',
                tags=['currency', 'exchange', 'conversion', 'forex'],
                examples=[
                    'What is the exchange rate from USD to EUR?',
                    'Convert 1000 USD to JPY',
                    'How much is 500 EUR in GBP?',
                ],
            )
        ],
    )
    
    activity_agent_card = AgentCard(
        name='Activity Planner Agent',
        url='http://localhost:10021',
        description='Plans activities and provides travel recommendations',
        version='1.0',
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        preferred_transport=TransportProtocol.jsonrpc,
        skills=[
            AgentSkill(
                id='activity_planning',
                name='Activity Planning',
                description='Create personalized travel itineraries and activity recommendations',
                tags=['travel', 'activities', 'itinerary', 'recommendations'],
                examples=[
                    'Plan a day trip in Paris',
                    'Recommend restaurants in Tokyo',
                    'What are the must-see attractions in Rome?',
                ],
            )
        ],
    )
    
    travel_manager_card = AgentCard(
        name='SK Travel Manager',
        url='http://localhost:10022',
        description='Comprehensive travel planning agent that orchestrates currency and activity services',
        version='1.0',
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        preferred_transport=TransportProtocol.jsonrpc,
        skills=[
            AgentSkill(
                id='comprehensive_travel_planning',
                name='Comprehensive Travel Planning',
                description='Handles all aspects of travel planning including currency and activities',
                tags=['travel', 'planning', 'currency', 'activities', 'orchestration'],
                examples=[
                    'Plan a budget-friendly trip to Seoul with currency exchange info',
                    'I need help with my Tokyo trip including money exchange and activities',
                    'What should I do in London and how much money should I exchange?',
                ],
            )
        ],
    )
    
    logger.debug('✅ Agent cards defined')
    
    """
    ## Create A2A Server Helper Function
    
    This function creates an A2A (Agent-to-Agent) protocol server by setting up HTTP communication infrastructure, configuring a request handler with task management and push notifications, wrapping everything in a Starlette web application, and returning the runnable server instance. 
    
    A2AStarletteApplication is a web application wrapper built on the Starlette ASGI framework that implements the A2A protocol standard, allowing AI agents to communicate with each other over HTTP using standardized message formats and discovery mechanisms.
    """
    logger.info("## Create A2A Server Helper Function")
    
    def create_a2a_server(agent_executor, agent_card):
        """Create an A2A server for any agent executor."""
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
    
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=BasePushNotificationSender(
                httpx_client, push_config_store),
        )
    
        app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
    
        if hasattr(app, 'build'):
            return app.build()
        elif hasattr(app, 'app'):
            return app.app
        else:
            return app
    
    
    logger.debug('✅ A2A server helper function created')
    
    """
    ## Start All A2A Servers
    
    We'll run all three agents as separate A2A servers using uvicorn.
    """
    logger.info("## Start All A2A Servers")
    
    async def run_agent_server(agent_executor, agent_card, port):
        """Run a single agent server with proper error handling."""
        try:
            app = create_a2a_server(agent_executor, agent_card)
    
            config = uvicorn.Config(
                app,
                host='127.0.0.1',
                port=port,
                log_level='info',
                loop='none',
                timeout_keep_alive=30,
                limit_concurrency=100,
            )
    
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"Error starting server on port {port}: {str(e)}")
            raise
    
    
    
    running_servers = []
    
    
    async def start_all_servers_background():
        """Start all servers in background tasks."""
        global running_servers
    
        try:
            currency_executor = CurrencyAgentExecutor()
            activity_executor = ActivityAgentExecutor()
            travel_executor = SemanticKernelTravelAgentExecutor()
    
            tasks = [
                asyncio.create_task(run_agent_server(
                    currency_executor, currency_agent_card, 10020)),
                asyncio.create_task(run_agent_server(
                    activity_executor, activity_agent_card, 10021)),
                asyncio.create_task(run_agent_server(
                    travel_executor, travel_manager_card, 10022)),
            ]
    
            running_servers = tasks
    
            await asyncio.sleep(3)
    
            logger.debug('✅ All A2A agent servers started in background!')
            logger.debug('   - Currency Exchange Agent: http://127.0.0.1:10020')
            logger.debug('   - Activity Planner Agent: http://127.0.0.1:10021')
            logger.debug('   - Travel Manager Agent: http://127.0.0.1:10022')
    
            return tasks
    
        except Exception as e:
            logger.error(f"Error in start_all_servers: {str(e)}")
            logger.debug(f"Failed to start servers: {str(e)}")
            raise
    
    server_tasks = await start_all_servers_background()
    logger.success(format_json(server_tasks))
    
    async def verify_servers():
        """Verify that all A2A servers are running and accessible."""
    
        servers = [
            ('Currency Exchange Agent', 'http://localhost:10020'),
            ('Activity Planner Agent', 'http://localhost:10021'),
            ('Travel Manager Agent', 'http://localhost:10022'),
        ]
    
        logger.debug("🔍 Verifying A2A servers...")
        logger.debug("="*50)
    
        async with httpx.AsyncClient() as client:
                for name, url in servers:
                    try:
                        response = await client.get(f"{url}{AGENT_CARD_WELL_KNOWN_PATH}", timeout=5.0)
                        if response.status_code == 200:
                            logger.debug(f"✅ {name} is running at {url}")
                        else:
                            logger.debug(f"⚠️ {name} returned status {response.status_code}")
                    except Exception as e:
                        logger.debug(f"❌ {name} is not accessible: {str(e)}")
            
        logger.success(format_json(result))
        logger.debug("="*50)
    
    await verify_servers()
    
    """
    ## Create A2A Client
    
    Now let's create a client to interact with our A2A agents.
    """
    logger.info("## Create A2A Client")
    
    class A2AClient:
        """Simple A2A client to interact with A2A servers."""
    
        def __init__(self, default_timeout: float = 60.0):
            self._agent_info_cache = {}
            self.default_timeout = default_timeout
    
        async def send_message(self, agent_url: str, message: str) -> str:
            """Send a message to an A2A agent."""
            timeout_config = httpx.Timeout(
                timeout=self.default_timeout,
                connect=10.0,
                read=self.default_timeout,
                write=10.0,
                pool=5.0,
            )
    
            async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
                    if agent_url not in self._agent_info_cache:
                        agent_card_response = await httpx_client.get(
                            f'{agent_url}{AGENT_CARD_WELL_KNOWN_PATH}'
                        )
                        self._agent_info_cache[agent_url] = agent_card_response.json()
                
                    agent_card_data = self._agent_info_cache[agent_url]
                    agent_card = AgentCard(**agent_card_data)
                
                    config = ClientConfig(
                        httpx_client=httpx_client,
                        supported_transports=[
                            TransportProtocol.jsonrpc,
                            TransportProtocol.http_json,
                        ],
                        use_client_preference=True,
                    )
                
                    factory = ClientFactory(config)
                    client = factory.create(agent_card)
                
                    message_obj = create_text_message_object(content=message)
                
                    responses = []
                    async for response in client.send_message(message_obj):
                        responses.append(response)
                
                    if responses:
                        try:
                            for response_item in responses:
                                if isinstance(response_item, tuple) and len(response_item) > 0:
                                    task = response_item[0]
                
                                    if hasattr(task, 'artifacts') and task.artifacts:
                                        for artifact in task.artifacts:
                                            if hasattr(artifact, 'parts') and artifact.parts:
                                                for part in artifact.parts:
                                                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                                        return part.root.text
                                                    elif hasattr(part, 'text'):
                                                        return part.text
                                                    elif hasattr(part, 'content'):
                                                        return str(part.content)
                                            elif hasattr(artifact, 'text'):
                                                return artifact.text
                                            elif hasattr(artifact, 'content'):
                                                return str(artifact.content)
                
                                    elif hasattr(task, 'text'):
                                        return task.text
                                    elif hasattr(task, 'content'):
                                        return str(task.content)
                                    elif hasattr(task, 'result'):
                                        return str(task.result)
                                    else:
                                        logger.debug(f"Debug - Task type: {type(task)}")
                                        logger.debug(f"Debug - Task attributes: {dir(task)}")
                                        if hasattr(task, '__dict__'):
                                            logger.debug(f"Debug - Task dict: {task.__dict__}")
                                        return f"Received response but couldn't parse content. Task: {str(task)}"
                                else:
                                    response_text = str(response_item)
                                    if response_text and response_text != "None":
                                        return response_text
                
                            return f"Received {len(responses)} responses but couldn't extract text content"
                
                        except Exception as e:
                            return f"Error parsing response: {str(e)}. Raw responses: {str(responses)}"
                
                    return 'No response received'
                
            logger.success(format_json(result))
    a2a_client = A2AClient()
    logger.debug('✅ A2A client updated with enhanced response parsing')
    
    """
    ## Test Individual Agents
    
    Let's test each agent individually to see how they work.
    """
    logger.info("## Test Individual Agents")
    
    async def test_currency_agent():
        """Test the currency exchange agent."""
        logger.debug("\n🔍 Testing Currency Exchange Agent")
        logger.debug("="*50)
    
        response = await a2a_client.send_message(
                'http://localhost:10020',
                'What is the exchange rate from USD to EUR and JPY?'
            )
        logger.success(format_json(response))
    
        logger.debug("User: What is the exchange rate from USD to EUR and JPY?")
        logger.debug("\nCurrency Agent Response:")
        logger.debug(response)
    
    await test_currency_agent()
    
    async def test_activity_agent():
        """Test the activity planner agent."""
        logger.debug("\n🔍 Testing Activity Planner Agent")
        logger.debug("="*50)
    
        response = await a2a_client.send_message(
                'http://localhost:10021',
                'Plan a one-day itinerary for Paris including must-see attractions'
            )
        logger.success(format_json(response))
    
        logger.debug("User: Plan a one-day itinerary for Paris including must-see attractions")
        logger.debug("\nActivity Agent Response:")
        logger.debug(response)
    
    await test_activity_agent()
    
    """
    ## Test the Travel Manager (Orchestrator)
    
    Now let's test the main Travel Manager agent that orchestrates the other agents.
    """
    logger.info("## Test the Travel Manager (Orchestrator)")
    
    async def test_travel_manager():
        """Test the travel manager orchestrating multiple agents."""
        logger.debug("\n🔍 Testing Travel Manager Agent (Orchestrator)")
        logger.debug("="*50)
    
        response = await a2a_client.send_message(
                'http://localhost:10022',
                'I am planning a trip to Tokyo. I have 1000 USD to exchange. What is the current exchange rate to JPY and what activities do you recommend for a 2-day visit?'
            )
        logger.success(format_json(response))
    
        logger.debug("User: I am planning a trip to Tokyo. I have 1000 USD to exchange.")
        logger.debug("      What is the current exchange rate to JPY and what activities")
        logger.debug("      do you recommend for a 2-day visit?")
        logger.debug("\nTravel Manager Response:")
        logger.debug(response)
    
    await test_travel_manager()
    
    """
    ## Interactive Testing
    
    Try your own travel planning queries!
    """
    logger.info("## Interactive Testing")
    
    async def interactive_test():
        """Interactive testing function."""
        logger.debug("\n🎯 Interactive Travel Planning Assistant")
        logger.debug("="*50)
        logger.debug("Available agents:")
        logger.debug("1. Currency Exchange Agent (port 10020) - Currency conversions")
        logger.debug("2. Activity Planner Agent (port 10021) - Travel recommendations")
        logger.debug("3. Travel Manager Agent (port 10022) - Comprehensive planning")
        logger.debug("\nExample queries:")
        logger.debug("- 'Convert 500 EUR to GBP'")
        logger.debug("- 'Plan a romantic dinner in Rome'")
        logger.debug("- 'I need help planning a budget trip to Seoul with 2000 USD'")
    
        user_query = "I'm visiting London next week with a budget of 1500 EUR. What's the exchange rate to GBP and what are the top attractions I should visit?"
    
        logger.debug(f"\nYour query: {user_query}")
        logger.debug("\nProcessing with Travel Manager...")
    
        response = await a2a_client.send_message(
                'http://localhost:10022',
                user_query
            )
        logger.success(format_json(response))
    
        logger.debug("\nResponse:")
        logger.debug(response)
    
    await interactive_test()
    
    """
    ## Summary
    
    Congratulations! You've successfully built a multi-agent travel planning system using:
    
    ### Technologies Used:
    - **Semantic Kernel** - For building intelligent agents
    - **Azure Ollama** - For LLM capabilities
    - **A2A Protocol** - For standardized agent communication
    - **Uvicorn** - For running local A2A servers
    
    ### What You've Learned:
    1. **Agent Creation**: Building specialized agents with Semantic Kernel
    2. **A2A Integration**: Wrapping SK agents for A2A protocol compatibility
    3. **Agent Orchestration**: Using a manager agent to coordinate multiple specialists
    4. **Real-time Services**: Integrating external APIs (Frankfurter for currency rates)
    5. **Local Deployment**: Running multiple A2A servers in a single notebook
    
    ### Next Steps:
    - Deploy agents to Azure Container Instances or Azure Functions
    - Add more specialized agents (flight booking, hotel recommendations)
    - Implement agent memory for context-aware conversations
    - Add authentication and security for production deployment
    - Create a web interface for the travel planning system
    
    ### Key Advantages of This Architecture:
    - **Modularity**: Each agent can be developed and deployed independently
    - **Scalability**: Agents can be scaled based on demand
    - **Reusability**: Agents can be reused in different applications
    - **Interoperability**: A2A protocol allows integration with agents from different frameworks
    """
    logger.info("## Summary")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())