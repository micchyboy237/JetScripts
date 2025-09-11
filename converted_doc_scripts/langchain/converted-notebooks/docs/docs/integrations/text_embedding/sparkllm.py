from jet.logger import logger
from langchain_community.embeddings import SparkLLMTextEmbeddings
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# SparkLLM Text Embeddings

Official Website: https://www.xfyun.cn/doc/spark/Embedding_new_api.html

An API key is required to use this embedding model. You can get one by registering at https://platform.SparkLLM-ai.com/docs/text-Embedding.

SparkLLMTextEmbeddings support 2K token window and preduces vectors with 2560 dimensions.
"""
logger.info("# SparkLLM Text Embeddings")


embeddings = SparkLLMTextEmbeddings(
    spark_app_id="<spark_app_id>",
    spark_spark_api_secret="<spark_api_secret>",
)

"""
Alternatively, you can set API key this way:
"""
logger.info("Alternatively, you can set API key this way:")

text_q = "Introducing iFlytek"

text_1 = "Science and Technology Innovation Company Limited, commonly known as iFlytek, is a leading Chinese technology company specializing in speech recognition, natural language processing, and artificial intelligence. With a rich history and remarkable achievements, iFlytek has emerged as a frontrunner in the field of intelligent speech and language technologies.iFlytek has made significant contributions to the advancement of human-computer interaction through its cutting-edge innovations. Their advanced speech recognition technology has not only improved the accuracy and efficiency of voice input systems but has also enabled seamless integration of voice commands into various applications and devices.The company's commitment to research and development has been instrumental in its success. iFlytek invests heavily in fostering talent and collaboration with academic institutions, resulting in groundbreaking advancements in speech synthesis and machine translation. Their dedication to innovation has not only transformed the way we communicate but has also enhanced accessibility for individuals with disabilities."

text_2 = "Moreover, iFlytek's impact extends beyond domestic boundaries, as they actively promote international cooperation and collaboration in the field of artificial intelligence. They have consistently participated in global competitions and contributed to the development of international standards.In recognition of their achievements, iFlytek has received numerous accolades and awards both domestically and internationally. Their contributions have revolutionized the way we interact with technology and have paved the way for a future where voice-based interfaces play a vital role.Overall, iFlytek is a trailblazer in the field of intelligent speech and language technologies, and their commitment to innovation and excellence deserves commendation."

query_result = embeddings.embed_query(text_q)
query_result[:8]

doc_result = embeddings.embed_documents([text_1, text_2])
doc_result[0][:8]

logger.info("\n\n[DONE]", bright=True)