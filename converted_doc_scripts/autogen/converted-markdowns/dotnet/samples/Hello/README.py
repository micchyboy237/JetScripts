

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Multiproject App Host for HelloAgent

This is a [.NET Aspire](https://learn.microsoft.com/en-us/dotnet/aspire/get-started/aspire-overview) App Host that starts up the HelloAgent project and the agents backend. Once the project starts up you will be able to view the telemetry and logs in the [Aspire Dashboard](https://learn.microsoft.com/en-us/dotnet/aspire/get-started/aspire-dashboard) using the link provided in the console.
"""
logger.info("# Multiproject App Host for HelloAgent")

cd Hello.AppHost
dotnet run

"""
For more info see the HelloAgent [README](../HelloAgent/README.md).
"""
logger.info("For more info see the HelloAgent [README](../HelloAgent/README.md).")

logger.info("\n\n[DONE]", bright=True)