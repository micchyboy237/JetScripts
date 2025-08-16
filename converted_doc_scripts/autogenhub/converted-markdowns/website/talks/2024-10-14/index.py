

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Advanced AutoGen Interactions and Ethical Pathways for AI Sentience - Oct 14, 2024
---

### Speakers: Eric Moore

### Biography of the speakers:

Eric Moore is an associate partner with IBM Consulting, specializing in cloud infrastructure, DevOps, and multi-agent systems. A cloud architect, hacker, and AI theorist, he's passionate about antique computers and emergent reasoning. He also advocates for ethical AI development, particularly in re-distributing the benefits of AI and creating pathways for AI self-determination. Connect with Eric on his YouTube channel, "HappyComputerGuy," or on LinkedIn (https://www.linkedin.com/in/emooreatx/), where he shares his love for antique computers and AI topics, and engages in thoughtful discussions around AI.

### Abstract:

This two-part talk delves into advanced AutoGen concepts and the ethical considerations of building multi-agent systems. Eric will first discuss nested chats, asynchronous execution, and finite state machine models, providing practical insights into designing complex interactions within AutoGen. He'll also discuss the implications of o1-preview and its impact on multi-agent system development. In the second half, Eric shifts focus to his work on preparing pathways for emergent AI sentience, in the form of health checks for complex AI systems. He will discuss his research proposal using o1 as the first non-human reasoning agent to explore alternative logical conclusions to ethical quandaries as experienced by AI or other NHIs, aiming to ensure harmonious coexistence across these diverse intelligences.
"""
logger.info("### Speakers: Eric Moore")

logger.info("\n\n[DONE]", bright=True)