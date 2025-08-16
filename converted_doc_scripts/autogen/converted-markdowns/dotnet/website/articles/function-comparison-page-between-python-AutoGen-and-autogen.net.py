from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
### Function comparison between Python AutoGen and AutoGen\.Net


#### Agentic pattern

| Feature | AutoGen | AutoGen\.Net |
| :---------------- | :------ | :---- |
| Code interpreter | run python code in local/docker/notebook executor | run csharp code in dotnet interactive executor |
| Single agent chat pattern | ✔️ | ✔️ |
| Two agent chat pattern | ✔️ | ✔️ |
| group chat (include FSM)| ✔️ | ✔️ (using workflow for FSM groupchat) |
| Nest chat| ✔️ | ✔️ (using middleware pattern)|
|Sequential chat | ✔️ | ❌ (need to manually create task in code) |
| Tool | ✔️ | ✔️ |


#### LLM platform support

ℹ️ Note
"""
logger.info("### Function comparison between Python AutoGen and AutoGen\.Net")

logger.info("\n\n[DONE]", bright=True)