async def main():
    from jet.transformers.formatters import format_json
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import TaskResult
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core import CancellationToken
    from autogen_core.models import UserMessage
    from autogen_ext.models.azure import AzureAIChatCompletionClient
    from azure.core.credentials import AzureKeyCredential
    from datasets import load_dataset
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from langfuse import Langfuse
    from langfuse import get_client
    import openlit
    import os
    import pandas as pd
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # AutoGen Agents in Production: Observability & Evaluation
    
    In this tutorial, we will learn how to **monitor the internal steps (traces) of [Autogen agents](https://github.com/microsoft/autogen)** and **evaluate its performance** using [Langfuse](https://langfuse.com).
    
    This guide covers **online** and **offline** evaluation metrics used by teams to bring agents to production fast and reliably. 
    
    **Why AI agent Evaluation is important:**
    - Debugging issues when tasks fail or produce suboptimal results
    - Monitoring costs and performance in real-time
    - Improving reliability and safety through continuous feedback
    
    ## Step 1: Set Environment Variables
    
    Get your Langfuse API keys by signing up for [Langfuse Cloud](https://cloud.langfuse.com/) or [self-hosting Langfuse](https://langfuse.com/self-hosting). 
    
    _**Note:** Self-hosters can use [Terraform modules](https://langfuse.com/self-hosting/azure) to deploy Langfuse on Azure. Alternatively, you can deploy Langfuse on Kubernetes using the [Helm chart](https://langfuse.com/self-hosting/kubernetes-helm)._
    """
    logger.info("# AutoGen Agents in Production: Observability & Evaluation")
    
    
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
    os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
    
    """
    With the environment variables set, we can now initialize the Langfuse client. `get_client()` initializes the Langfuse client using the credentials provided in the environment variables.
    """
    logger.info("With the environment variables set, we can now initialize the Langfuse client. `get_client()` initializes the Langfuse client using the credentials provided in the environment variables.")
    
    
    langfuse = Langfuse(
        blocked_instrumentation_scopes=["autogen SingleThreadedAgentRuntime"]
    )
    
    if langfuse.auth_check():
        logger.debug("Langfuse client is authenticated and ready!")
    else:
        logger.debug("Authentication failed. Please check your credentials and host.")
    
    """
    ## Step 2: Initialize OpenLit Instrumentation
    
    Now, we initialize the [OpenLit](https://github.com/openlit/openlit) instrumentation. OpenLit automatically captures AutoGen operations and exports OpenTelemetry (OTel) spans to Langfuse.
    """
    logger.info("## Step 2: Initialize OpenLit Instrumentation")
    
    
    openlit.init(tracer=langfuse._otel_tracer, disable_batch=True, disabled_instrumentors=["mistral"])
    
    """
    ## Step 3: Run your agent
    
    Now we set up a multi turn agent to test our instrumentation.
    """
    logger.info("## Step 3: Run your agent")
    
    
    
    
    client = AzureAIChatCompletionClient(
        model="llama3.2",
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
        model_info={
            "json_output": True,
            "function_calling": True,
            "vision": True,
            "family": "unknown",
            "structured_output": False
        },
    )
    
    meal_planner_agent = AssistantAgent(
        "meal_planner_agent",
        model_client=client,
        description="A seasoned meal-planning coach who suggests balanced meals.",
        system_message="""
        You are a Meal-Planning Assistant with a decade of experience helping busy people prepare meals.
        Goal: propose the single best meal (breakfast, lunch, or dinner) given the user's context.
        Each response must contain ONLY one complete meal idea (title + very brief component list) — no extras.
        Keep it concise: skip greetings, chit-chat, and filler.
        """,
    )
    
    nutritionist_agent = AssistantAgent(
        "nutritionist_agent",
        model_client=client,
        description="A registered dietitian ensuring meals meet nutritional standards.",
        system_message="""
        You are a Nutritionist focused on whole-food, macro-balanced eating.
        Evaluate the meal_planner_agent’s recommendation.
        If the meal is nutritionally sound, sufficiently varied, and portion-appropriate, respond with 'APPROVE'.
        Otherwise, give high-level guidance on how to improve it (e.g. 'add a plant-based protein') — do NOT provide a full alternative recipe.
        """,
    )
    
    termination = TextMentionTermination("APPROVE")
    
    team = RoundRobinGroupChat(
        [meal_planner_agent, nutritionist_agent],
        termination_condition=termination,
    )
    
    user_input = "I'm looking for a quick, delicious dinner I can prep after work. I have 30 minutes and minimal clean-up is ideal."
    
    with langfuse.start_as_current_span(name="create_meal_plan") as span:
        async for message in team.run_stream(task=user_input):
            if isinstance(message, TaskResult):
                logger.debug("Stop Reason:", message.stop_reason)
            else:
                logger.debug(message)
    
        span.update_trace(
            input=user_input,
            output=message.stop_reason,
        )
    
    langfuse.flush()
    
    """
    ### Trace Structure
    
    Langfuse records a **trace** that contains **spans**, which represent each step of your agent’s logic. Here, the trace contains the overall agent run and sub-spans for:
    - The meal planner agent
    - The nuritionist agents
    
    You can inspect these to see precisely where time is spent, how many tokens are used, and so on:
    
    ![Trace tree in Langfuse](https://langfuse.com/images/cookbook/example-autogen-evaluation/trace-tree.png)
    
    _[Link to the trace](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/dac2b33e7cd709e685ccf86a137ecc64)_
    
    ## Online Evaluation
    
    Online Evaluation refers to evaluating the agent in a live, real-world environment, i.e. during actual usage in production. This involves monitoring the agent’s performance on real user interactions and analyzing outcomes continuously.
    
    ### Common Metrics to Track in Production
    
    1. **Costs** — The instrumentation captures token usage, which you can transform into approximate costs by assigning a price per token.
    2. **Latency** — Observe the time it takes to complete each step, or the entire run.
    3. **User Feedback** — Users can provide direct feedback (thumbs up/down) to help refine or correct the agent.
    4. **LLM-as-a-Judge** — Use a separate LLM to evaluate your agent’s output in near real-time (e.g., checking for toxicity or correctness).
    
    Below, we show examples of these metrics.
    
    #### 1. Costs
    
    Below is a screenshot showing usage for `llama3.2` calls. This is useful to see costly steps and optimize your agent.
    
    ![Costs](https://langfuse.com/images/cookbook/example-autogen-evaluation/gpt-4o-costs.png) 
    
    _[Link to the trace](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/dac2b33e7cd709e685ccf86a137ecc64)_
    
    #### 2. Latency
    
    We can also see how long it took to complete each step. In the example below, the entire run took about 3 seconds, which you can break down by step. This helps you identify bottlenecks and optimize your agent.
    
    ![Latency](https://langfuse.com/images/cookbook/example-autogen-evaluation/agent-latency.png) 
    
    _[Link to the trace](https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/dac2b33e7cd709e685ccf86a137ecc64?display=timeline)_
    
    #### 3. User Feedback
    
    If your agent is embedded into a user interface, you can record direct user feedback (like a thumbs-up/down in a chat UI).
    """
    logger.info("### Trace Structure")
    
    
    langfuse = get_client()
    
    with langfuse.start_as_current_span(
        name="autogen-request-user-feedback-1") as span:
    
        async for message in team.run_stream(task="Create a meal with potatoes"):
                if isinstance(message, TaskResult):
                    logger.debug("Stop Reason:", message.stop_reason)
                else:
                    logger.debug(message)
    
        span.score_trace(
            name="user-feedback",
            value=1,
            data_type="NUMERIC",
            comment="This was delicious, thank you"
        )
    
    with langfuse.start_as_current_span(name="autogen-request-user-feedback-2") as span:
    
        async for message in team.run_stream(task="I am allergic to gluten."):
                if isinstance(message, TaskResult):
                    logger.debug("Stop Reason:", message.stop_reason)
                else:
                    logger.debug(message)
    
        langfuse.score_current_trace(
            name="user-feedback",
            value=1,
            data_type="NUMERIC"
        )
    
    langfuse.create_score(
        trace_id="predefined_trace_id",
        name="user-feedback",
        value=1,
        data_type="NUMERIC",
        comment="This was correct, thank you"
    )
    
    """
    User feedback is then captured in Langfuse:
    
    ![User feedback is being captured in Langfuse](https://langfuse.com/images/cookbook/example-autogen-evaluation/user-feedback.png)
    
    #### 4. Automated LLM-as-a-Judge Scoring
    
    LLM-as-a-Judge is another way to automatically evaluate your agent's output. You can set up a separate LLM call to gauge the output’s correctness, toxicity, style, or any other criteria you care about.
    
    **Workflow**:
    1. You define an **Evaluation Template**, e.g., "Check if the text is toxic."
    2. You set a model that is used as judge-model; in this case `llama3.2` queried via Azure.
    2. Each time your agent generates output, you pass that output to your "judge" LLM with the template.
    3. The judge LLM responds with a rating or label that you log to your observability tool.
    
    Example from Langfuse:
    
    ![LLM-as-a-Judge Evaluator](https://langfuse.com/images/cookbook/example-autogen-evaluation/evaluator.png)
    """
    logger.info("#### 4. Automated LLM-as-a-Judge Scoring")
    
    with langfuse.start_as_current_span(name="autogen-request-user-feedback-2") as span:
    
        async for message in team.run_stream(task="I am a picky eater and not sure if you find something for me."):
                if isinstance(message, TaskResult):
                    logger.debug("Stop Reason:", message.stop_reason)
                else:
                    logger.debug(message)
    
        span.update_trace(
            input=user_input,
            output=message.stop_reason,
        )
    
    langfuse.flush()
    
    """
    You can see that the answer of this example is judged as "not toxic".
    
    ![LLM-as-a-Judge Evaluation Score](https://langfuse.com/images/cookbook/example-autogen-evaluation/llm-as-a-judge-score.png)
    
    #### 5. Observability Metrics Overview
    
    All of these metrics can be visualized together in dashboards. This enables you to quickly see how your agent performs across many sessions and helps you to track quality metrics over time.
    
    ![Observability metrics overview](https://langfuse.com/images/cookbook/example-autogen-evaluation/dashboard.png)
    
    ## Offline Evaluation
    
    Online evaluation is essential for live feedback, but you also need **offline evaluation**—systematic checks before or during development. This helps maintain quality and reliability before rolling changes into production.
    
    ### Dataset Evaluation
    
    In offline evaluation, you typically:
    1. Have a benchmark dataset (with prompt and expected output pairs)
    2. Run your agent on that dataset
    3. Compare outputs to the expected results or use an additional scoring mechanism
    
    Below, we demonstrate this approach with the [q&a-dataset](https://huggingface.co/datasets/junzhang1207/search-dataset), which contains questions and expected answers.
    """
    logger.info("#### 5. Observability Metrics Overview")
    
    
    dataset = load_dataset("junzhang1207/search-dataset", split = "train")
    df = pd.DataFrame(dataset)
    logger.debug("First few rows of search-dataset:")
    logger.debug(df.head())
    
    """
    Next, we create a dataset entity in Langfuse to track the runs. Then, we add each item from the dataset to the system.
    """
    logger.info("Next, we create a dataset entity in Langfuse to track the runs. Then, we add each item from the dataset to the system.")
    
    langfuse = Langfuse()
    
    langfuse_dataset_name = "qa-dataset_autogen-agent"
    
    langfuse.create_dataset(
        name=langfuse_dataset_name,
        description="q&a dataset uploaded from Hugging Face",
        metadata={
            "date": "2025-03-21",
            "type": "benchmark"
        }
    )
    
    df_25 = df.sample(25) # For this example, we upload only 25 dataset questions
    
    for idx, row in df_25.iterrows():
        langfuse.create_dataset_item(
            dataset_name=langfuse_dataset_name,
            input={"text": row["question"]},
            expected_output={"text": row["expected_answer"]}
        )
    
    """
    ![Dataset items in Langfuse](https://langfuse.com/images/cookbook/example-autogen-evaluation/example-dataset.png)
    
    #### Running the Agent on the Dataset
    
    First, we assemble a simple Autogen agent that answers questions using Azure Ollama models.
    """
    logger.info("#### Running the Agent on the Dataset")
    
    
    
    load_dotenv()
    client = AzureAIChatCompletionClient(
        model="llama3.2", log_dir=f"{LOG_DIR}/chats",
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
        max_tokens=5000,
        model_info={
            "json_output": True,
            "function_calling": False,
            "vision": False,
            "family": "unknown",
            "structured_output": True,
        },
    )
    
    result = await client.create([UserMessage(content="What is the capital of France?", source="user")])
    logger.success(format_json(result))
    logger.debug(result)
    
    agent = AssistantAgent(
        name="assistant",
        model_client=client,
        tools=[],
        system_message="You are participant in a quizz show and you are given a question. You need to create a short answer to the question.",
    )
    
    """
    Then, we define a helper function `my_agent()`.
    """
    logger.info("Then, we define a helper function `my_agent()`.")
    
    async def my_agent(user_query: str):
    
        with langfuse.start_as_current_span(name="autogen-trace") as span:
    
            response = await agent.on_messages(
                    [TextMessage(content=user_query, source="user")],
                    cancellation_token=CancellationToken(),
                )
            logger.success(format_json(response))
    
            span.update_trace(
                input=user_query,
                output=response.chat_message.content,
            )
    
        return str(response.chat_message.content)
    
    await my_agent("What is the capital of France?")
    
    """
    Finally, we loop over each dataset item, run the agent, and link the trace to the dataset item. We can also attach a quick evaluation score if desired.
    """
    logger.info("Finally, we loop over each dataset item, run the agent, and link the trace to the dataset item. We can also attach a quick evaluation score if desired.")
    
    dataset_name = "qa-dataset_autogen-agent"
    current_run_name = "dev_tasks_run-autogen_gpt-4.1" # Identifies this specific evaluation run
    current_run_metadata={"model_provider": "Azure", "model": "gpt-4.1"}
    current_run_description="Evaluation run for Autogen model on July 3rd"
    
    dataset = langfuse.get_dataset('qa-dataset_autogen-agent')
    
    for item in dataset.items:
        logger.debug(f"Running evaluation for item: {item.id} (Input: {item.input})")
    
        with item.run(
            run_name=current_run_name,
            run_metadata=current_run_metadata,
            run_description=current_run_description
        ) as root_span:
            generated_answer = await my_agent(user_query = item.input["text"])
            logger.success(format_json(generated_answer))
    
        logger.debug("Generated Answer: ", generated_answer)
    
    logger.debug(f"\nFinished processing dataset '{dataset_name}' for run '{current_run_name}'.")
    
    langfuse.flush()
    
    """
    You can repeat this process with different agent configurations such as:
    - Models (llama3.2, gpt-4.1, etc.)
    - Prompts
    - Tools (search vs. no search)
    - Complexity of agent (multi agent vs single agent)
    
    Then compare them side-by-side in Langfuse. In this example, I did run the agent 3 times on the 25 dataset questions. For each run, I used a different Azure Ollama model. You can see that amount of correctly answered questions improves when using a larger model (as expected). The `correct_answer` score is created by an [LLM-as-a-Judge Evaluator](https://langfuse.com/docs/scores/model-based-evals) that is set up to judge the correctness of the question based on the sample answer given in the dataset.
    
    ![Dataset run overview](https://langfuse.com/images/cookbook/example-autogen-evaluation/dataset_runs.png)
    ![Dataset run comparison](https://langfuse.com/images/cookbook/example-autogen-evaluation/dataset-run-comparison.png)
    """
    logger.info("You can repeat this process with different agent configurations such as:")
    
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