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
title: 'Development'
description: 'Learn how to preview changes locally'
---

<Info>
  **Prerequisite** You should have installed Node.js (version 18.10.0 or
  higher).
</Info>

Step 1. Install Mintlify on your OS:

<CodeGroup>
"""
logger.info("title: 'Development'")

npm i -g mintlify

"""

"""

yarn global add mintlify

"""
</CodeGroup>

Step 2. Go to the docs are located (where you can find `mint.json`) and run the following command:
"""
logger.info("Step 2. Go to the docs are located (where you can find `mint.json`) and run the following command:")

mintlify dev

"""
The documentation website is now available at `http://localhost:3000`.

### Custom Ports

Mintlify uses port 3000 by default. You can use the `--port` flag to customize the port Mintlify runs on. For example, use this command to run in port 3333:
"""
logger.info("### Custom Ports")

mintlify dev --port 3333

"""
You will see an error like this if you try to run Mintlify in a port that's already taken:
"""
logger.info("You will see an error like this if you try to run Mintlify in a port that's already taken:")

Error: listen EADDRINUSE: address already in use :::3000

"""
## Mintlify Versions

Each CLI is linked to a specific version of Mintlify. Please update the CLI if your local website looks different than production.

<CodeGroup>
"""
logger.info("## Mintlify Versions")

npm i -g mintlify@latest

"""

"""

yarn global upgrade mintlify

"""
</CodeGroup>

## Deployment

<Tip>
  Unlimited editors available under the [Startup
  Plan](https://mintlify.com/pricing)
</Tip>

You should see the following if the deploy successfully went through:

<Frame>
  <img src="/images/checks-passed.png" style={{ borderRadius: '0.5rem' }} />
</Frame>

## Troubleshooting

Here's how to solve some common problems when working with the CLI.

<AccordionGroup>
  <Accordion title="Mintlify is not loading">
    Update to Node v18. Run `mintlify install` and try again.
  </Accordion>
  <Accordion title="No such file or directory on Windows">
Go to the `C:/Users/Username/.mintlify/` directory and remove the `mint`
folder. Then Open the Git Bash in this location and run `git clone
https://github.com/mintlify/mint.git`.

Repeat step 3.

  </Accordion>
  <Accordion title="Getting an unknown error">
    Try navigating to the root of your device and delete the ~/.mintlify folder.
    Then run `mintlify dev` again.
  </Accordion>
</AccordionGroup>

Curious about what changed in a CLI version? [Check out the CLI changelog.](/changelog/command-line)
"""
logger.info("## Deployment")

logger.info("\n\n[DONE]", bright=True)