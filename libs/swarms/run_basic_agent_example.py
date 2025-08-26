import os
import shutil
from jet.file.utils import save_file
from jet.libs.swarms.jet_examples.swarm_examples_1 import basic_agent_example

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

response = basic_agent_example()
save_file(response, f"{OUTPUT_DIR}/response.md")
