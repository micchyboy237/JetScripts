

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
Agents for running the [AgentEval](https://autogenhub.github.io/autogen/blog/2023/11/20/AgentEval/) pipeline.

AgentEval is a process for evaluating a LLM-based system's performance on a given task.

When given a task to evaluate and a few example runs, the critic and subcritic agents create evaluation criteria for evaluating a system's solution. Once the criteria has been created, the quantifier agent can evaluate subsequent task solutions based on the generated criteria.

For more information see: [AgentEval Integration Roadmap](https://github.com/microsoft/autogen/issues/2162)

See our [blog post](https://autogenhub.github.io/autogen/blog/2024/06/21/AgentEval) for usage examples and general explanations.
"""
logger.info("Agents for running the [AgentEval](https://autogenhub.github.io/autogen/blog/2023/11/20/AgentEval/) pipeline.")

logger.info("\n\n[DONE]", bright=True)