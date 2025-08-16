from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Microsoft Fabric

![Fabric Example](img/ecosystem-fabric.png)

[Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/get-started/microsoft-fabric-overview) is an all-in-one analytics solution for enterprises that covers everything from data movement to data science, Real-Time Analytics, and business intelligence. It offers a comprehensive suite of services, including data lake, data engineering, and data integration, all in one place. In this notenook, we give a simple example for using AutoGen in Microsoft Fabric.

- [Microsoft Fabric + AutoGen Code Examples](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_microsoft_fabric.ipynb)
"""
logger.info("# Microsoft Fabric")

logger.info("\n\n[DONE]", bright=True)