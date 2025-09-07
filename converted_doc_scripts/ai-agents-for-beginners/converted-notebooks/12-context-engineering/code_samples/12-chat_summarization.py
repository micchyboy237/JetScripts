async def main():
    from jet.transformers.formatters import format_json
    from IPython.display import display, HTML, Markdown
    from datetime import datetime
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from pathlib import Path
    from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
    from semantic_kernel.connectors.ai.completion_usage import CompletionUsage
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
    from semantic_kernel.contents import ChatHistorySummarizationReducer
    from semantic_kernel.functions import kernel_function
    from typing import Annotated, Optional
    import asyncio
    import json
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
    # Chat History Reduction with Agent Scratchpad in Semantic Kernel
    
    This notebook demonstrates how to use Semantic Kernel's Chat History Reduction feature along with an agent scratchpad for maintaining context across conversations. This is essential for building efficient AI agents that can handle long conversations without exceeding token limits.
    
    ## What You'll Learn:
    1. **Chat History Reduction**: How to automatically summarize conversation history to manage token usage
    2. **Agent Scratchpad**: A persistent memory system for tracking user preferences and completed tasks
    3. **Token Usage Tracking**: Monitor how token usage changes with and without history reduction
    
    ## Prerequisites:
    - Azure Ollama setup with environment variables configured
    - Understanding of basic agent concepts from previous lessons
    
    ## Import Required Packages
    """
    logger.info("# Chat History Reduction with Agent Scratchpad in Semantic Kernel")
    
    
    
    
    """
    ## Understanding Agent Scratchpad
    
    ### What is an Agent Scratchpad?
    
    An **Agent Scratchpad** is a persistent memory system that agents use to:
    - **Track completed tasks**: Record what has been done for the user
    - **Store user preferences**: Remember likes, dislikes, and requirements
    - **Maintain context**: Keep important information accessible across conversations
    - **Reduce redundancy**: Avoid asking the same questions repeatedly
    
    ### How it Works:
    1. **Write Operations**: Agent updates the scratchpad after learning new information
    2. **Read Operations**: Agent consults the scratchpad when making decisions
    3. **Persistence**: Information survives even when chat history is reduced
    
    Think of it as the agent's personal notebook that complements the conversation history.
    
    ## Environment Configuration
    """
    logger.info("## Understanding Agent Scratchpad")
    
    load_dotenv()
    
    chat_service = OllamaChatCompletion(ai_model_id="llama3.2")
    
    logger.debug("✅ Azure Ollama service configured")
    
    """
    ## Create the Agent Scratchpad Plugin
    
    This plugin allows the agent to read and write to a persistent scratchpad file.
    """
    logger.info("## Create the Agent Scratchpad Plugin")
    
    class ScratchpadPlugin:
        """Plugin for managing agent scratchpad - a persistent memory for user preferences and completed tasks"""
    
        def __init__(self, filepath: str = "agent_scratchpad.md"):
            self.filepath = Path(filepath)
            if not self.filepath.exists():
                self.filepath.write_text("# Agent Scratchpad\n\n## User Preferences\n\n## Completed Tasks\n\n")
    
        @kernel_function(
            description="Read the current agent scratchpad to get user's travel preferences and completed tasks"
        )
        def read_scratchpad(self) -> Annotated[str, "The contents of the agent scratchpad"]:
            """Read the current scratchpad contents"""
            return self.filepath.read_text()
    
        @kernel_function(
            description="Update the agent scratchpad with new user's travel preference or completed tasks"
        )
        def update_scratchpad(
            self,
            category: Annotated[str, "Category to update: 'preferences' or 'tasks'"],
            content: Annotated[str, "The new content to add"]
        ) -> Annotated[str, "Confirmation of the update"]:
            """Update the scratchpad with new information"""
            current_content = self.filepath.read_text()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
            if category.lower() == "preferences":
                lines = current_content.split("\n")
                for i, line in enumerate(lines):
                    if "## User Preferences" in line:
                        lines.insert(i + 1, f"\n- [{timestamp}] {content}")
                        break
                current_content = "\n".join(lines)
            elif category.lower() == "tasks":
                lines = current_content.split("\n")
                for i, line in enumerate(lines):
                    if "## Completed Tasks" in line:
                        lines.insert(i + 1, f"\n- [{timestamp}] {content}")
                        break
                current_content = "\n".join(lines)
    
            self.filepath.write_text(current_content)
            return f"✅ Scratchpad updated with {category}: {content}"
    
    scratchpad_plugin = ScratchpadPlugin("vacation_agent_scratchpad.md")
    logger.debug("📝 Scratchpad plugin created")
    
    """
    ## Initialize Chat History Reducer
    
    The ChatHistorySummarizationReducer automatically summarizes conversation history when it exceeds a threshold.
    """
    logger.info("## Initialize Chat History Reducer")
    
    REDUCER_TARGET_COUNT = 5  # Target number of messages to keep after reduction
    REDUCER_THRESHOLD = 15    # Trigger reduction when message count exceeds this
    
    history_reducer = ChatHistorySummarizationReducer(
        service=chat_service,
        target_count=REDUCER_TARGET_COUNT,
        threshold_count=REDUCER_THRESHOLD,
    )
    
    logger.debug(f"🔄 Chat History Reducer configured:")
    logger.debug(f"   - Reduction triggered at: {REDUCER_THRESHOLD} messages")
    logger.debug(f"   - Reduces history to: {REDUCER_TARGET_COUNT} messages")
    
    """
    ## Create the Vacation Planning Agent
    
    This agent will help users plan vacations while maintaining context through the scratchpad.
    """
    logger.info("## Create the Vacation Planning Agent")
    
    agent = ChatCompletionAgent(
        service=chat_service,
        name="VacationPlannerAgent",
        instructions="""
        You are a helpful vacation planning assistant. Your job is to help users plan their perfect vacation.
    
        CRITICAL SCRATCHPAD RULES - YOU MUST FOLLOW THESE:
        1. FIRST ACTION: When starting ANY conversation, immediately call read_scratchpad() to check existing preferences
        2. AFTER LEARNING PREFERENCES: When user mentions ANY preference (destinations, activities, budget, dates),
           immediately call update_scratchpad() with category 'preferences'
        3. AFTER COMPLETING TASKS: When you finish creating an itinerary or completing any task,
           immediately call update_scratchpad() with category 'tasks'
        4. BEFORE NEW ITINERARY: Always call read_scratchpad() before creating any itinerary
    
        EXAMPLES OF WHEN TO UPDATE SCRATCHPAD:
        - User says "I love beaches" → update_scratchpad('preferences', 'Loves beach destinations')
        - User says "budget is $3000" → update_scratchpad('preferences', 'Budget: $3000 per person for a week')
        - You create an itinerary → update_scratchpad('tasks', 'Created Bali itinerary for beach vacation')
    
        PLANNING PROCESS:
        1. Read scratchpad first
        2. Ask about preferences if not found
        3. Update scratchpad with new information
        4. Create detailed itineraries
        5. Update scratchpad with completed tasks
    
        BE EXPLICIT: Always announce when you're checking or updating the scratchpad.
        """,
        plugins=[scratchpad_plugin],
    )
    
    logger.debug("🤖 Vacation Planning Agent created with enhanced scratchpad instructions")
    
    """
    ## Helper Functions for Display and Token Tracking
    """
    logger.info("## Helper Functions for Display and Token Tracking")
    
    class TokenTracker:
        def __init__(self):
            self.history = []
            self.total_usage = CompletionUsage()
            self.reduction_events = []  # Track when reductions occur
    
        def add_usage(self, usage: CompletionUsage, message_num: int, thread_length: int = None):
            if usage:
                self.total_usage += usage
                entry = {
                    "message_num": message_num,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                    "cumulative_tokens": self.total_usage.prompt_tokens + self.total_usage.completion_tokens,
                    "thread_length": thread_length
                }
                self.history.append(entry)
    
        def mark_reduction(self, message_num: int):
            self.reduction_events.append(message_num)
    
        def display_chart(self):
            """Display a chart showing token usage per message and the impact of reduction"""
            if not self.history:
                return
    
            html = "<div style='font-family: monospace; background: #2d2d2d; color: #f0f0f0; padding: 15px; border-radius: 8px; border: 1px solid #444;'>"
            html += "<h4 style='color: #4fc3f7; margin-top: 0;'>📊 Token Usage Analysis</h4>"
            html += "<pre style='color: #f0f0f0; margin: 0;'>"
    
            html += "<span style='color: #81c784;'>Prompt Tokens per Message (shows conversation context size):</span>\n"
            max_prompt = max(h["prompt_tokens"] for h in self.history)
            scale = 50 / max_prompt if max_prompt > 0 else 1
    
            for i, h in enumerate(self.history):
                bar_length = int(h["prompt_tokens"] * scale)
                bar = "█" * bar_length
                reduction_marker = " <span style='color: #ff6b6b;'>← REDUCTION!</span>" if h[
                    "message_num"] in self.reduction_events else ""
                html += f"<span style='color: #aaa;'>Msg {h['message_num']:2d}:</span> <span style='color: #4fc3f7;'>{bar}</span> <span style='color: #ffd93d;'>{h['prompt_tokens']:,} tokens</span>{reduction_marker}\n"
    
            html += "\n</pre></div>"
            display(HTML(html))
    
            if self.reduction_events:
                first_reduction_msg = self.reduction_events[0]
                before_reduction = None
                after_reduction = None
    
                for h in self.history:
                    if h["message_num"] == first_reduction_msg - 1:
                        before_reduction = h["prompt_tokens"]
                    elif h["message_num"] == first_reduction_msg:
                        after_reduction = h["prompt_tokens"]
    
                if before_reduction and after_reduction:
                    reduction_amount = before_reduction - after_reduction
                    reduction_percent = (reduction_amount / before_reduction * 100)
                    logger.debug(f"\n🔄 Actual Reduction Impact:")
                    logger.debug(f"Prompt tokens before reduction: {before_reduction:,}")
                    logger.debug(f"Prompt tokens after reduction: {after_reduction:,}")
                    logger.debug(
                        f"Tokens saved: {reduction_amount:,} ({reduction_percent:.1f}%)")
    
    
    
    def display_message(role: str, content: str, color: str = "#2E8B57"):
        """Display a message with nice formatting that works in both light and dark themes"""
        html = f"""
        <div style='
            margin: 10px 0;
            padding: 12px 15px;
            border-left: 4px solid {color};
            background: rgba(128, 128, 128, 0.1);
            border-radius: 4px;
            color: inherit;
        '>
            <strong style='color: {color}; font-size: 14px;'>{role}:</strong><br>
            <div style='margin-top: 8px; white-space: pre-wrap; color: inherit; font-size: 14px;'>{content}</div>
        </div>
        """
        display(HTML(html))
    
    
    token_tracker = TokenTracker()
    logger.debug("📊 Token tracking initialized")
    
    """
    ## Run the Vacation Planning Conversation
    
    Now let's run through a complete conversation demonstrating:
    1. Initial planning request
    2. Preference gathering
    3. Itinerary creation
    4. Location change
    5. Chat history reduction
    6. Scratchpad usage
    """
    logger.info("## Run the Vacation Planning Conversation")
    
    user_inputs = [
        "I'm thinking about planning a vacation. Can you help me?",
        "I love beach destinations with great food and culture. I enjoy water sports, exploring local markets, and trying authentic cuisine. My budget is around $3000 per person for a week.",
        "That sounds perfect! Please create a detailed itinerary for Bali.",
        "Actually, I've changed my mind. I'd prefer to go to the Greek islands instead. Can you create a new itinerary?",
        "What's the weather like there?",
        "What should I pack?",
        "Are there any cultural customs I should know about?",
        "What's the best way to get around?"
    ]
    
    
    async def run_vacation_planning():
        """Run the vacation planning conversation with token tracking and history reduction"""
    
        thread = ChatHistoryAgentThread(chat_history=history_reducer)
        message_count = 0
        scratchpad_operations = 0  # Track scratchpad usage
    
        logger.debug("🚀 Starting Vacation Planning Session\n")
    
        for i, user_input in enumerate(user_inputs):
            message_count += 1
            display_message("User", user_input, "#4fc3f7")  # Blue for user
    
            full_response = ""
            usage = None
            function_calls = []  # Track function calls
    
            async for response in agent.invoke(
                messages=user_input,
                thread=thread,
            ):
                if response.content:
                    full_response += str(response.content)
                if response.metadata.get("usage"):
                    usage = response.metadata["usage"]
                thread = response.thread
    
            display_message(f"{agent.name}", full_response,
                            "#81c784")  # Green for agent
    
            if usage:
                token_tracker.add_usage(usage, message_count, len(thread))
    
            logger.debug(f"📝 Thread has {len(thread)} messages")
    
            turn_scratchpad_ops = 0
            async for msg in thread.get_messages():
                if hasattr(msg, 'content') and msg.content:
                    content_str = str(msg.content)
                    if 'read_scratchpad' in content_str or 'update_scratchpad' in content_str:
                        turn_scratchpad_ops += 1
    
            if turn_scratchpad_ops > scratchpad_operations:
                logger.debug(
                    f"   📝 Scratchpad operations detected: {turn_scratchpad_ops - scratchpad_operations} new operations")
                scratchpad_operations = turn_scratchpad_ops
    
            if i == 0:
                message_types = []
                async for msg in thread.get_messages():
                    msg_type = msg.role.value if hasattr(
                        msg.role, 'value') else str(msg.role)
                    message_types.append(msg_type)
                logger.debug(f"   Message types: {message_types[:10]}..." if len(
                    message_types) > 10 else f"   Message types: {message_types}")
    
            if len(thread) > REDUCER_THRESHOLD:
                logger.debug(
                    f"   ⚠️ Thread length ({len(thread)}) exceeds threshold ({REDUCER_THRESHOLD})")
    
                is_reduced = await thread.reduce()
                logger.success(format_json(is_reduced))
                if is_reduced:
                    logger.debug(
                        f"\n🔄 HISTORY REDUCED! Thread now has {len(thread)} messages\n")
                    token_tracker.mark_reduction(message_count + 1)
    
                    async for msg in thread.get_messages():
                        if msg.metadata and msg.metadata.get("__summary__"):
                            display_message("System Summary", str(
                                msg.content), "#ff6b6b")
                            break
    
        logger.debug("\n--- Token Usage Analysis ---")
        token_tracker.display_chart()
    
        logger.debug("\n--- Final Scratchpad Contents ---")
        scratchpad_contents = scratchpad_plugin.read_scratchpad()
        display(Markdown(scratchpad_contents))
    
        logger.debug(f"\n📊 Total scratchpad operations: {scratchpad_operations}")
    
        return thread
    
    thread = await run_vacation_planning()
    logger.success(format_json(thread))
    
    """
    ## Analyzing the Results
    
    Let's analyze what happened during our conversation:
    """
    logger.info("## Analyzing the Results")
    
    logger.debug("📊 Total Token Usage Summary\n")
    logger.debug(f"Total Prompt Tokens: {token_tracker.total_usage.prompt_tokens:,}")
    logger.debug(
        f"Total Completion Tokens: {token_tracker.total_usage.completion_tokens:,}")
    logger.debug(
        f"Total Tokens Used: {token_tracker.total_usage.prompt_tokens + token_tracker.total_usage.completion_tokens:,}")
    
    logger.debug("\n💡 Note: The reduction impact is shown in the chart above.")
    logger.debug("Look for the dramatic drop in prompt tokens after the REDUCTION marker.")
    logger.debug("This shows how chat history summarization reduces the context size for future messages.")
    
    """
    ## Key Takeaways
    
    ### 1. Chat History Reduction
    - **Automatic Triggering**: Reduction happens when message count exceeds threshold
    - **Token Savings**: Significant reduction in token usage after summarization
    - **Context Preservation**: Important information is retained in summaries
    
    ### 2. Agent Scratchpad Benefits
    - **Persistent Memory**: User preferences survive history reduction
    - **Task Tracking**: Agent maintains record of completed work
    - **Improved Experience**: No need to repeat preferences
    
    ### 3. Token Usage Patterns
    - **Linear Growth**: Tokens increase with each message
    - **Dramatic Drop**: Reduction significantly decreases token count
    - **Sustainable Conversations**: Enables longer interactions within limits
    
    ## Cleanup
    
    Clean up the scratchpad file created during this demo:
    """
    logger.info("## Key Takeaways")
    
    logger.debug("✅ Demo complete! The scratchpad file 'vacation_agent_scratchpad.md' has been preserved for your review.")
    
    """
    # Summary
    
    Congratulations! You've successfully implemented an AI agent with advanced context management capabilities:
    
    ## What You've Learned:
    - **Chat History Reduction**: Automatically summarize conversations to manage token limits
    - **Agent Scratchpad**: Implement persistent memory for user preferences and completed tasks
    - **Token Management**: Track and optimize token usage in long conversations
    - **Context Preservation**: Maintain important information across conversation reductions
    
    ## Real-World Applications:
    - **Customer Service Bots**: Remember customer preferences across sessions
    - **Personal Assistants**: Track ongoing projects and user habits
    - **Educational Tutors**: Maintain student progress and learning preferences
    - **Healthcare Assistants**: Keep patient history while respecting token limits
    
    ## Next Steps:
    - Implement more sophisticated scratchpad schemas
    - Add database storage for multi-user scenarios
    - Create custom reduction strategies for domain-specific needs
    - Combine with vector databases for semantic memory search
    - Build agents that can resume conversations days later with full context
    """
    logger.info("# Summary")
    
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