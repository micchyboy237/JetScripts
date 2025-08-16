from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
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