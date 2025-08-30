async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms import ChatMessage, LLM
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.tools.types import BaseTool
    from llama_index.core.workflow import (
        Context,
        Workflow,
        StartEvent,
        StopEvent,
        step,
    )
    from llama_index.core.workflow import Context
    from llama_index.core.workflow import Event
    from typing import Any, Callable, List
    from typing import Any, Dict, Tuple
    import ast
    import contextlib
    import inspect
    import io
    import os
    import re
    import shutil
    import traceback

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    LOG_DIR = f"{OUTPUT_DIR}/logs"

    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    ##
    
    # Creating a CodeAct Agent From Scratch
    
    While LlamaIndex provides a pre-built [CodeActAgent](https://docs.llamaindex.ai/en/stable/examples/agent/code_act_agent/), we can also create our own from scratch.
    
    This way, we can fully understand and customize the agent's behaviour beyond what is provided by the pre-built agent.
    
    In this notebook, we will
    1. Create a workflow for generating and parsing code
    2. Implement basic code execution
    3. Add memory and state to the agent
    
    ## Setting up Functions for our Agent
    
    If we want our agent to execute our code, we need to deine the code for it to execute!
    
    For now, let's use a few basic math functions.
    """
    logger.info("##")

    def add(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b

    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    def divide(a: int, b: int) -> float:
        """Divide two numbers"""
        return a / b

    """
    ## Creating a Code Executor 
    
    In order to execute code, we need to create a code executor.
    
    Here, we will use a simple in-process code executor that maintains it's own state.
    
    **NOTE:** This is a simple example, and does not include proper sandboxing. In a production environment, you should use tools like docker or proper code sandboxing environments.
    """
    logger.info("## Creating a Code Executor")

    class SimpleCodeExecutor:
        """
        A simple code executor that runs Python code with state persistence.
        This executor maintains a global and local state between executions,
        allowing for variables to persist across multiple code runs.
        NOTE: not safe for production use! Use with caution.
        """

        def __init__(self, locals: Dict[str, Any], globals: Dict[str, Any]):
            """
            Initialize the code executor.
            Args:
                locals: Local variables to use in the execution context
                globals: Global variables to use in the execution context
            """
            self.globals = globals
            self.locals = locals

        def execute(self, code: str) -> Tuple[bool, str, Any]:
            """
            Execute Python code and capture output and return values.
            Args:
                code: Python code to execute
            Returns:
                Tuple of (success, output, return_value)
            """
            logger.debug(f"Executing code: {code}")
            logger.debug(f"Available locals: {list(self.locals.keys())}")
            stdout = io.StringIO()
            stderr = io.StringIO()
            output = ""
            return_value = None
            success = False
            try:
                # Validate code for undefined variables
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        if node.id not in self.locals and node.id not in self.globals and node.id not in __builtins__.__dict__:
                            output = f"Error: NameError: name '{node.id}' is not defined"
                            return False, output, None

                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec_globals = self.globals.copy()
                    exec_locals = self.locals.copy()
                    try:
                        last_node = tree.body[-1] if tree.body else None
                        if isinstance(last_node, ast.Expr):
                            last_line = code.rstrip().split("\n")[-1]
                            exec_code = code[: -len(last_line)] + \
                                "\n__result__ = " + last_line
                            exec(exec_code, exec_globals, exec_locals)
                            return_value = exec_locals.get("__result__")
                        else:
                            exec(code, exec_globals, exec_locals)
                        self.locals.update(exec_locals)
                        success = True
                    except Exception as e:
                        output = f"Error: {type(e).__name__}: {str(e)}\n"
                        output += traceback.format_exc()
                output += stdout.getvalue()
                if stderr.getvalue():
                    output += "\n" + stderr.getvalue()
            except Exception as e:
                output = f"Error: {type(e).__name__}: {str(e)}\n"
                output += traceback.format_exc()
            if return_value is not None:
                output += "\n\n" + str(return_value)
            logger.debug(
                f"Execution result: success={success}, output={output}, return_value={return_value}")
            return success, output, return_value

    code_executor = SimpleCodeExecutor(
        locals={
            "add": add,
            "subtract": subtract,
            "multiply": multiply,
            "divide": divide,
        },
        globals={
            "__builtins__": __builtins__,
            "np": __import__("numpy"),
        },
    )

    """
    ## Defining the CodeAct Agent
    
    Now, we can using LlamaIndex Workflows to define the workflow for our agent.
    
    The basic flow is:
    - take in our prompt + chat history
    - parse out the code to execute (if any)
    - execute the code
    - provide the output of the code execution back to the agent
    - repeat until the agent is satisfied with the answer
    
    First, we can create the events in the workflow.
    """
    logger.info("## Defining the CodeAct Agent")

    class InputEvent(Event):
        input: list[ChatMessage]

    class StreamEvent(Event):
        delta: str

    class CodeExecutionEvent(Event):
        code: str

    """
    Next, we can define the workflow that orchestrates using these events.
    """
    logger.info(
        "Next, we can define the workflow that orchestrates using these events.")

    CODEACT_SYSTEM_PROMPT = """
    You are a helpful assistant that can execute code.
    
    Given the chat history, you can write code within <execute>...</execute> tags to help the user with their question.
    
    In your code, you can reference any previously used variables or functions.
    
    The user has also provided you with some predefined functions:
    {fn_str}
    
    To execute code, write the code between <execute>...</execute> tags.
    """

    class CodeActAgent(Workflow):
        def __init__(
            self,
            fns: List[Callable],
            code_execute_fn: Callable,
            llm: LLM | None = None,
            **workflow_kwargs: Any,
        ) -> None:
            # Increase timeout to 120 seconds
            super().__init__(timeout=120.0, **workflow_kwargs)
            self.fns = fns or []
            self.code_execute_fn = code_execute_fn
            self.llm = llm or OllamaFunctionCallingAdapter(model="llama3.2")
            self.fn_str = "\n\n".join(
                f'def {fn.__name__}{str(inspect.signature(fn))}:\n    """ {fn.__doc__} """\n    ...'
                for fn in self.fns
            )
            self.system_message = ChatMessage(
                role="system",
                content=CODEACT_SYSTEM_PROMPT.format(fn_str=self.fn_str),
            )

        def _parse_code(self, response: str) -> str | None:
            matches = re.findall(
                r"<execute>(.*?)</execute>", response, re.DOTALL)
            if matches:
                return "\n\n".join(matches)
            return None

        @step
        async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
            memory = await ctx.store.get("memory", default=None)
            logger.success(format_json(memory))
            if not memory:
                memory = ChatMemoryBuffer.from_defaults(llm=self.llm)
            user_input = ev.get("user_input")
            if user_input is None:
                raise ValueError("user_input kwarg is required")
            user_msg = ChatMessage(role="user", content=user_input)
            memory.put(user_msg)
            chat_history = memory.get()
            await ctx.store.set("memory", memory)
            return InputEvent(input=[self.system_message, *chat_history])

        @step
        async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> CodeExecutionEvent | StopEvent:
            chat_history = ev.input
            response = await self.llm.achat(chat_history)
            ctx.write_event_to_stream(StreamEvent(
                delta=response.message.content or ""))
            memory = await ctx.store.get("memory")
            logger.success(format_json(memory))
            memory.put(response.message)
            await ctx.store.set("memory", memory)
            code = self._parse_code(response.message.content)
            if not code:
                return StopEvent(result=response)
            else:
                return CodeExecutionEvent(code=code)

        @step
        async def handle_code_execution(self, ctx: Context, ev: CodeExecutionEvent) -> InputEvent:
            ctx.write_event_to_stream(ev)
            success, output, return_value = self.code_execute_fn(ev.code)
            memory = await ctx.store.get("memory")
            logger.success(format_json(memory))
            content = output
            if success and return_value is not None:
                content = f"{output}\nResult: {return_value}"
            memory.put(ChatMessage(role="assistant", content=content))
            await ctx.store.set("memory", memory)
            chat_history = memory.get()
            return InputEvent(input=[self.system_message, *chat_history])

    """
    ## Testing the CodeAct Agent
    
    Now, we can test out the CodeAct Agent!
    
    We'll create a simple agent and slowly build up the complexity with requests.
    """
    logger.info("## Testing the CodeAct Agent")

    agent = CodeActAgent(
        fns=[add, subtract, multiply, divide],
        code_execute_fn=code_executor.execute,
        llm=OllamaFunctionCallingAdapter(
            model="llama3.2", log_dir=f"{LOG_DIR}/chats"),
    )

    ctx = Context(agent)

    async def run_agent_verbose(agent: CodeActAgent, ctx: Context, query: str):
        handler = agent.run(user_input=query, ctx=ctx)
        logger.debug(f"User:  {query}")
        async for event in handler.stream_events():
            if isinstance(event, StreamEvent):
                logger.debug(f"{event.delta}", end="", flush=True)
            elif isinstance(event, CodeExecutionEvent):
                logger.debug(f"\n-----------\nParsed code:\n{event.code}\n")

        return await handler

    response = await run_agent_verbose(
        agent, ctx, "Calculate the sum of all numbers from 1 to 10"
    )
    logger.success(format_json(response))

    response = await run_agent_verbose(
        agent, ctx, "Add 5 and 3, then multiply the result by 2"
    )
    logger.success(format_json(response))

    response = await run_agent_verbose(
        agent, ctx, "Calculate the sum of the first 10 fibonacci numbers0"
    )
    logger.success(format_json(response))

    response = await run_agent_verbose(
        agent, ctx, "Calculate the sum of the first 20 fibonacci numbers"
    )
    logger.success(format_json(response))

    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
