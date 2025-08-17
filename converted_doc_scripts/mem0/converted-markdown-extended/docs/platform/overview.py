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
---
title: Overview
description: 'Empower your AI applications with long-term memory and personalization'
icon: "eye"
iconType: "solid"
---

## Welcome to Mem0 Platform

The Mem0 Platform is a managed service and the easiest way to add our powerful memory layer to your applications. 

## Why Choose Mem0 Platform?

Mem0 Platform offers a powerful, user-centric solution for AI memory management with a few key features:

1. **Simplified Development**: Integrate comprehensive memory capabilities with just 4 lines of code. Our API-first approach allows you to focus on building great features while we handle the complexities of memory management.

2. **Scalable Solution**: Whether you're working on a prototype or a production-ready system, Mem0 is designed to grow with your application. Our platform effortlessly scales to meet your evolving needs.

3. **Enhanced Performance**: Experience lightning-fast response times with sub-50ms latency, ensuring smooth and responsive user interactions in your AI applications.

4. **Insightful Dashboard**: Gain valuable insights and maintain full control over your AI's memory through our intuitive dashboard. Easily manage memories and access key user insights.


## Getting Started

Check out our [Platform Guide](/platform/quickstart) to start using Mem0 platform quickly.

## Next Steps

- Sign up to the [Mem0 Platform](https://mem0.dev/pd)
- Join our [Discord](https://mem0.dev/Did) with other developers and get support.

We're excited to see what you'll build with Mem0 Platform. Let's create smarter, more personalized AI experiences together!
"""
logger.info("## Welcome to Mem0 Platform")

logger.info("\n\n[DONE]", bright=True)