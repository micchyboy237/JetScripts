async def main():
    from jet.transformers.formatters import format_json
    from autogen_core import (
    AgentId,
    ClosureAgent,
    ClosureContext,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    default_subscription,
    message_handler,
    type_subscription,
    )
    from dataclasses import dataclass
    from jet.logger import CustomLogger
    import asyncio
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Concurrent Agents
    
    In this section, we explore the use of multiple agents working concurrently. We cover three main patterns:
    
    1. **Single Message & Multiple Processors**  
       Demonstrates how a single message can be processed by multiple agents subscribed to the same topic simultaneously.
    
    2. **Multiple Messages & Multiple Processors**  
       Illustrates how specific message types can be routed to dedicated agents based on topics.
    
    3. **Direct Messaging**  
       Focuses on sending messages between agents and from the runtime to agents.
    """
    logger.info("# Concurrent Agents")
    
    
    
    @dataclass
    class Task:
        task_id: str
    
    
    @dataclass
    class TaskResponse:
        task_id: str
        result: str
    
    """
    ## Single Message & Multiple Processors
    The first pattern shows how a single message can be processed by multiple agents simultaneously:
    
    - Each `Processor` agent subscribes to the default topic using the {py:meth}`~autogen_core.components.default_subscription` decorator.
    - When publishing a message to the default topic, all registered agents will process the message independently.
    ```{note}
    Below, we are subscribing `Processor` using the {py:meth}`~autogen_core.components.default_subscription` decorator, there's an alternative way to subscribe an agent without using decorators altogether as shown in [Subscribe and Publish to Topics](../framework/message-and-communication.ipynb#subscribe-and-publish-to-topics), this way the same agent class can be subscribed to different topics.
    ```
    """
    logger.info("## Single Message & Multiple Processors")
    
    @default_subscription
    class Processor(RoutedAgent):
        @message_handler
        async def on_task(self, message: Task, ctx: MessageContext) -> None:
            logger.debug(f"{self._description} starting task {message.task_id}")
            await asyncio.sleep(2)  # Simulate work
            logger.debug(f"{self._description} finished task {message.task_id}")
    
    runtime = SingleThreadedAgentRuntime()
    
    await Processor.register(runtime, "agent_1", lambda: Processor("Agent 1"))
    await Processor.register(runtime, "agent_2", lambda: Processor("Agent 2"))
    
    runtime.start()
    
    await runtime.publish_message(Task(task_id="task-1"), topic_id=DefaultTopicId())
    
    await runtime.stop_when_idle()
    
    """
    ## Multiple messages & Multiple Processors
    Second, this pattern demonstrates routing different types of messages to specific processors:
    - `UrgentProcessor` subscribes to the "urgent" topic
    - `NormalProcessor` subscribes to the "normal" topic
    
    We make an agent subscribe to a specific topic type using the {py:meth}`~autogen_core.components.type_subscription` decorator.
    """
    logger.info("## Multiple messages & Multiple Processors")
    
    TASK_RESULTS_TOPIC_TYPE = "task-results"
    task_results_topic_id = TopicId(type=TASK_RESULTS_TOPIC_TYPE, source="default")
    
    
    @type_subscription(topic_type="urgent")
    class UrgentProcessor(RoutedAgent):
        @message_handler
        async def on_task(self, message: Task, ctx: MessageContext) -> None:
            logger.debug(f"Urgent processor starting task {message.task_id}")
            await asyncio.sleep(1)  # Simulate work
            logger.debug(f"Urgent processor finished task {message.task_id}")
    
            task_response = TaskResponse(task_id=message.task_id, result="Results by Urgent Processor")
            await self.publish_message(task_response, topic_id=task_results_topic_id)
    
    
    @type_subscription(topic_type="normal")
    class NormalProcessor(RoutedAgent):
        @message_handler
        async def on_task(self, message: Task, ctx: MessageContext) -> None:
            logger.debug(f"Normal processor starting task {message.task_id}")
            await asyncio.sleep(3)  # Simulate work
            logger.debug(f"Normal processor finished task {message.task_id}")
    
            task_response = TaskResponse(task_id=message.task_id, result="Results by Normal Processor")
            await self.publish_message(task_response, topic_id=task_results_topic_id)
    
    """
    After registering the agents, we can publish messages to the "urgent" and "normal" topics:
    """
    logger.info("After registering the agents, we can publish messages to the "urgent" and "normal" topics:")
    
    runtime = SingleThreadedAgentRuntime()
    
    await UrgentProcessor.register(runtime, "urgent_processor", lambda: UrgentProcessor("Urgent Processor"))
    await NormalProcessor.register(runtime, "normal_processor", lambda: NormalProcessor("Normal Processor"))
    
    runtime.start()
    
    await runtime.publish_message(Task(task_id="normal-1"), topic_id=TopicId(type="normal", source="default"))
    await runtime.publish_message(Task(task_id="urgent-1"), topic_id=TopicId(type="urgent", source="default"))
    
    await runtime.stop_when_idle()
    
    """
    #### Collecting Results
    
    In the previous example, we relied on console printing to verify task completion. However, in real applications, we typically want to collect and process the results programmatically.
    
    To collect these messages, we'll use a {py:class}`~autogen_core.components.ClosureAgent`. We've defined a dedicated topic `TASK_RESULTS_TOPIC_TYPE` where both `UrgentProcessor` and `NormalProcessor` publish their results. The ClosureAgent will then process messages from this topic.
    """
    logger.info("#### Collecting Results")
    
    queue = asyncio.Queue[TaskResponse]()
    
    
    async def collect_result(_agent: ClosureContext, message: TaskResponse, ctx: MessageContext) -> None:
        await queue.put(message)
    
    
    runtime.start()
    
    CLOSURE_AGENT_TYPE = "collect_result_agent"
    await ClosureAgent.register_closure(
        runtime,
        CLOSURE_AGENT_TYPE,
        collect_result,
        subscriptions=lambda: [TypeSubscription(topic_type=TASK_RESULTS_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)],
    )
    
    await runtime.publish_message(Task(task_id="normal-1"), topic_id=TopicId(type="normal", source="default"))
    await runtime.publish_message(Task(task_id="urgent-1"), topic_id=TopicId(type="urgent", source="default"))
    
    await runtime.stop_when_idle()
    
    while not queue.empty():
        logger.debug(await queue.get())
    
    """
    ## Direct Messages
    
    In contrast to the previous patterns, this pattern focuses on direct messages. Here we demonstrate two ways to send them:
    
    - Direct messaging between agents  
    - Sending messages from the runtime to specific agents  
    
    Things to consider in the example below:
    
    - Messages are addressed using the {py:class}`~autogen_core.components.AgentId`.  
    - The sender can expect to receive a response from the target agent.  
    - We register the `WorkerAgent` class only once; however, we send tasks to two different workers.
        - How? As stated in [Agent lifecycle](../core-concepts/agent-identity-and-lifecycle.md#agent-lifecycle), when delivering a message using an {py:class}`~autogen_core.components.AgentId`, the runtime will either fetch the instance or create one if it doesn't exist. In this case, the runtime creates two instances of workers when sending those two messages.
    """
    logger.info("## Direct Messages")
    
    class WorkerAgent(RoutedAgent):
        @message_handler
        async def on_task(self, message: Task, ctx: MessageContext) -> TaskResponse:
            logger.debug(f"{self.id} starting task {message.task_id}")
            await asyncio.sleep(2)  # Simulate work
            logger.debug(f"{self.id} finished task {message.task_id}")
            return TaskResponse(task_id=message.task_id, result=f"Results by {self.id}")
    
    
    class DelegatorAgent(RoutedAgent):
        def __init__(self, description: str, worker_type: str):
            super().__init__(description)
            self.worker_instances = [AgentId(worker_type, f"{worker_type}-1"), AgentId(worker_type, f"{worker_type}-2")]
    
        @message_handler
        async def on_task(self, message: Task, ctx: MessageContext) -> TaskResponse:
            logger.debug(f"Delegator received task {message.task_id}.")
    
            subtask1 = Task(task_id="task-part-1")
            subtask2 = Task(task_id="task-part-2")
    
            worker1_result, worker2_result = await asyncio.gather(
                    self.send_message(subtask1, self.worker_instances[0]), self.send_message(subtask2, self.worker_instances[1])
                )
            logger.success(format_json(worker1_result, worker2_result))
    
            combined_result = f"Part 1: {worker1_result.result}, " f"Part 2: {worker2_result.result}"
            task_response = TaskResponse(task_id=message.task_id, result=combined_result)
            return task_response
    
    runtime = SingleThreadedAgentRuntime()
    
    await WorkerAgent.register(runtime, "worker", lambda: WorkerAgent("Worker Agent"))
    await DelegatorAgent.register(runtime, "delegator", lambda: DelegatorAgent("Delegator Agent", "worker"))
    
    runtime.start()
    
    delegator = AgentId("delegator", "default")
    response = await runtime.send_message(Task(task_id="main-task"), recipient=delegator)
    logger.success(format_json(response))
    
    logger.debug(f"Final result: {response.result}")
    await runtime.stop_when_idle()
    
    """
    ## Additional Resources
    
    If you're interested in more about concurrent processing, check out the [Mixture of Agents](./mixture-of-agents.ipynb) pattern, which relies heavily on concurrent agents.
    """
    logger.info("## Additional Resources")
    
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