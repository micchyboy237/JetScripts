import sys
import io
from jet.logger import logger

# Redirect stdout to a buffer
buffer = io.StringIO()
sys.stdout = buffer

# Add some unwanted output
print("Unwanted output")

# Clear the buffer
buffer.seek(0)
buffer.truncate(0)

# Log something
logger.debug("Test logger")

# Add the desired output
print("export SAMPLE_ARG='final_output2'")

# Restore real stdout and send the correct output
sys.stdout = sys.__stdout__
print(buffer.getvalue().strip())
