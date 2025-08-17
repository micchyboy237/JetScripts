import os

from jet.cache.redis.data_utils import check_redis_data
from jet.logger import CustomLogger

# Set up logging
OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
log_file = os.path.join(OUTPUT_DIR, "redis_verify.log")
logger = CustomLogger(log_file, overwrite=True)

if __name__ == "__main__":
    check_redis_data()
