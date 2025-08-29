async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.llms import LLM, ChatMessage, TextBlock, ImageBlock
    from llama_index.core.memory import Memory
    from llama_index.core.memory import Memory, StaticMemoryBlock
    from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    )
    from pydantic import Field
    from typing import List, Literal, Optional
    import os
    import re
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # Manipulating Memory at Runtime
    
    In this notebook, we cover how to use the `Memory` class to build an agentic workflow with dynamic memory.
    
    Specifically, we will build a workflow where a user can upload a file, and pin that to the context of the LLM (i.e. like the file context in Cursor).
    
    By default, as the short-term memory fills up and is flushed, it will be passed to memory blocks for processing as needed (extracting facts, indexing for retrieval, or for static blocks, ignoring it).
    
    With this notebook, the intent is to show how memory can be managed and manipulated at runtime, beyond the already existing functionality described above.
    
    ## Setup
    
    For our workflow, we will use OllamaFunctionCallingAdapter as our LLM.
    """
    logger.info("# Manipulating Memory at Runtime")
    
    # !pip install llama-index-core llama-index-llms-ollama
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    """
    ## Workflow Setup
    
    Our workflow will be fairly straightfoward. There will be two main entry points
    
    1. Adding/Removing files from memory 
    2. Chatting with the LLM
    
    Using the `Memory` class, we can introduce memory blocks that hold our static context.
    """
    logger.info("## Workflow Setup")
    
    
    
    class InitEvent(StartEvent):
        user_msg: str
        new_file_paths: List[str] = Field(default_factory=list)
        removed_file_paths: List[str] = Field(default_factory=list)
    
    
    class ContextUpdateEvent(Event):
        new_file_paths: List[str] = Field(default_factory=list)
        removed_file_paths: List[str] = Field(default_factory=list)
    
    
    class ChatEvent(Event):
        pass
    
    
    class ResponseEvent(StopEvent):
        response: str
    
    
    class ContextualLLMChat(Workflow):
        def __init__(self, memory: Memory, llm: LLM, **workflow_kwargs):
            super().__init__(**workflow_kwargs)
            self._memory = memory
            self._llm = llm
    
        def _path_to_block_name(self, file_path: str) -> str:
            return re.sub(r"[^\w-]", "_", file_path)
    
        @step
        async def init(self, ev: InitEvent) -> ContextUpdateEvent | ChatEvent:
            await self._memory.aput(ChatMessage(role="user", content=ev.user_msg))
    
            if ev.new_file_paths or ev.removed_file_paths:
                return ContextUpdateEvent(
                    new_file_paths=ev.new_file_paths,
                    removed_file_paths=ev.removed_file_paths,
                )
            else:
                return ChatEvent()
    
        @step
        async def update_memory_context(self, ev: ContextUpdateEvent) -> ChatEvent:
            current_blocks = self._memory.memory_blocks
            current_block_names = [block.name for block in current_blocks]
    
            for new_file_path in ev.new_file_paths:
                if new_file_path not in current_block_names:
                    if new_file_path.endswith((".png", ".jpg", ".jpeg")):
                        self._memory.memory_blocks.append(
                            StaticMemoryBlock(
                                name=self._path_to_block_name(new_file_path),
                                static_content=[ImageBlock(path=new_file_path)],
                            )
                        )
                    elif new_file_path.endswith((".txt", ".md", ".py", ".ipynb")):
                        with open(new_file_path, "r") as f:
                            self._memory.memory_blocks.append(
                                StaticMemoryBlock(
                                    name=self._path_to_block_name(new_file_path),
                                    static_content=f.read(),
                                )
                            )
                    else:
                        raise ValueError(f"Unsupported file: {new_file_path}")
            for removed_file_path in ev.removed_file_paths:
                named_block = self._path_to_block_name(removed_file_path)
                self._memory.memory_blocks = [
                    block
                    for block in self._memory.memory_blocks
                    if block.name != named_block
                ]
    
            return ChatEvent()
    
        @step
        async def chat(self, ev: ChatEvent) -> ResponseEvent:
            chat_history = await self._memory.aget()
            logger.success(format_json(chat_history))
            response = self._llm.chat(chat_history)
            logger.success(format_json(response))
            return ResponseEvent(response=response.message.content)
    
    """
    ## Using the Workflow
    
    Now that we have our chat workflow defined, we can try it out! You can use any file, but for this example, we will use a few dummy files.
    """
    logger.info("## Using the Workflow")
    
    # !wget https://mediaproxy.tvtropes.org/width/1200/https://static.tvtropes.org/pmwiki/pub/images/shrek_cover.png -O ./image.png
    # !wget https://raw.githubusercontent.com/run-llama/llama_index/refs/heads/main/llama-index-core/llama_index/core/memory/memory.py -O ./memory.py
    
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)
    
    memory = Memory.from_defaults(
        session_id="my_session",
        token_limit=60000,
        chat_history_token_ratio=0.7,
        token_flush_size=5000,
        insert_method="user",
    )
    
    workflow = ContextualLLMChat(
        memory=memory,
        llm=llm,
        verbose=True,
    )
    
    """
    We can simulate a user adding a file to memory, and then chatting with the LLM.
    """
    logger.info("We can simulate a user adding a file to memory, and then chatting with the LLM.")
    
    response = await workflow.run(
            user_msg="What does this file contain?",
            new_file_paths=["./memory.py"],
        )
    logger.success(format_json(response))
    
    logger.debug("--------------------------------")
    logger.debug(response.response)
    
    """
    Great! Now, we can simulate a user removing that file, and adding a new one.
    """
    logger.info("Great! Now, we can simulate a user removing that file, and adding a new one.")
    
    response = await workflow.run(
            user_msg="What does this next file contain?",
            new_file_paths=["./image.png"],
            removed_file_paths=["./memory.py"],
        )
    logger.success(format_json(response))
    
    logger.debug("--------------------------------")
    logger.debug(response.response)
    
    """
    It works! Now, you've learned how to manage memory in a custom workflow. Beyond just letting short-term memory flush into memory blocks, you can manually manipulate the memory blocks at runtime as well.
    """
    logger.info("It works! Now, you've learned how to manage memory in a custom workflow. Beyond just letting short-term memory flush into memory blocks, you can manually manipulate the memory blocks at runtime as well.")
    
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