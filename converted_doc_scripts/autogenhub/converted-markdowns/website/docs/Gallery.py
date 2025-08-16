from jet.logger import CustomLogger
import GalleryPage from '../src/components/GalleryPage';
import galleryData from "../src/data/gallery.json";
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
hide_table_of_contents: true
---


# Gallery

This page contains a list of demos that use AutoGen in various applications from the community.

**Contribution guide:**
Built something interesting with AutoGen? Submit a PR to add it to the list! See the [Contribution Guide below](#contributing) for more details.

<GalleryPage items={galleryData} />

## Contributing

To contribute, please open a PR that adds an entry to the `data/gallery.json` file in the `src` directory. The entry should be an object with the following properties:
"""
logger.info("# Gallery")

{
    "title": "AutoGen Playground",
    "link": "https://huggingface.co/spaces/thinkall/AutoGen_Playground",
    "description": "A space to explore the capabilities of AutoGen.",
    "image": "default.png",
    "tags": ["ui"]
  }

"""
The `image` property should be the name of a file in the `static/img/gallery` directory.
The `tags` property should be an array of strings that describe the demo. We recommend using no more than two tags for clarity.
Here are the meanings of several tags for reference:
1. app: Using Autogen for specific applications.
2. extension: Enhancing AutoGen beyond the features in current version.
3. ui: Building user interface for AutoGen.
4. tool: Strengthening AutoGen Agents with external tools.
5. groupchat: Solving complex tasks with a group of Agents.

if the existing ones do not precisely portray your own demos, new tags are also encouraged to add.
"""
logger.info("The `image` property should be the name of a file in the `static/img/gallery` directory.")

logger.info("\n\n[DONE]", bright=True)