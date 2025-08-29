from PIL import Image
from jet.logger import CustomLogger
from llama_index.core.llama_pack import download_llama_pack
import matplotlib.pyplot as plt
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Amazon Product Extraction Pack

This LlamaPack provides an example of our Amazon Product Extraction pack.
"""
logger.info("# Amazon Product Extraction Pack")

# import nest_asyncio

# nest_asyncio.apply()


AmazonProductExtractionPack = download_llama_pack(
    "AmazonProductExtractionPack",
    "./amazon_product_extraction_pack",
    llama_hub_url="https://raw.githubusercontent.com/run-llama/llama-hub/jerry/add_amazon_product_extraction/llama_hub",
)

amazon_pack = AmazonProductExtractionPack(
    "https://www.amazon.com/AutoFocus-Microphone-NexiGo-Streaming-Compatible/dp/B08931JJLV/ref=sr_1_1_sspa?crid=ZXMK53A5VVNZ&keywords=webcams&qid=1701156679&sprefix=webcam%2Caps%2C147&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"
)

response = amazon_pack.run()

display(response)


imageUrl = "tmp.png"
image = Image.open(imageUrl).convert("RGB")
plt.figure(figsize=(16, 5))
plt.imshow(image)

logger.info("\n\n[DONE]", bright=True)