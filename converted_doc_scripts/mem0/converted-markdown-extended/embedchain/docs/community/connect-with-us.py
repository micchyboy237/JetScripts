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
title: 🤝 Connect with Us
---

We believe in building a vibrant and supportive community around embedchain. There are various channels through which you can connect with us, stay updated, and contribute to the ongoing discussions:

<CardGroup cols={3}>
  <Card title="Twitter" icon="twitter" href="https://twitter.com/embedchain">
    Follow us on Twitter
  </Card>
  <Card title="Slack" icon="slack" href="https://embedchain.ai/slack" color="#4A154B">
    Join our slack community
  </Card>
  <Card title="Discord" icon="discord" href="https://discord.gg/6PzXDgEjG5" color="#7289DA">
    Join our discord community
  </Card>
  <Card title="LinkedIn" icon="linkedin" href="https://www.linkedin.com/company/embedchain/">
  Connect with us on LinkedIn
  </Card>
  <Card title="Schedule a call" icon="calendar" href="https://cal.com/taranjeetio/ec">
  Schedule a call with Embedchain founder
  </Card>
  <Card title="Newsletter" icon="message" href="https://embedchain.substack.com/">
  Subscribe to our newsletter
  </Card>
</CardGroup>

We look forward to connecting with you and seeing how we can create amazing things together!
"""
logger.info("title: 🤝 Connect with Us")

logger.info("\n\n[DONE]", bright=True)