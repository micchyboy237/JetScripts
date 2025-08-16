

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
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