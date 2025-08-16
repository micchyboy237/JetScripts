from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
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