from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# NuGet Directory

This directory contains resources and metadata for packaging the AutoGen.NET SDK as a NuGet package.

## Files

- **icon.png**: The icon used for the NuGet package.
- **NUGET.md**: The readme file displayed on the NuGet package page.
- **NUGET-PACKAGE.PROPS**: The MSBuild properties file that defines the packaging settings for the NuGet package.

## Purpose

The files in this directory are used to configure and build the NuGet package for the AutoGen.NET SDK, ensuring that it includes necessary metadata, documentation, and resources.
"""
logger.info("# NuGet Directory")

logger.info("\n\n[DONE]", bright=True)