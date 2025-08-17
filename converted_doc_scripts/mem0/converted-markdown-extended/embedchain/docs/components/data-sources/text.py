from embedchain import App
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
title: '📝 Text'
---

### Text

Text is a local data type. To supply your own text, use the data_type as `text` and enter a string. The text is not processed, this can be very versatile. Eg:
"""
logger.info("### Text")


app = App()

app.add('Seek wealth, not money or status. Wealth is having assets that earn while you sleep. Money is how we transfer time and wealth. Status is your place in the social hierarchy.', data_type='text')

"""
Note: This is not used in the examples because in most cases you will supply a whole paragraph or file, which did not fit.
"""
logger.info("Note: This is not used in the examples because in most cases you will supply a whole paragraph or file, which did not fit.")

logger.info("\n\n[DONE]", bright=True)