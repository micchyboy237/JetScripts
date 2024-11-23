import sys
import io
from jet.logger import logger

# Temporarily redirect stdout
sys.stdout = io.StringIO()

# Log something
logger.debug("Test logger")

# This output is captured and won't go to real stdout
print('echo "Previous line"')

# Restore real stdout
sys.stdout = sys.__stdout__

# Only this will be sent to stdout and picked by eval
print("export SAMPLE_ARG='final_output1'")
