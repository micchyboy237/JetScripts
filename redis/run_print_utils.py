from jet.cache.redis import RedisPrintUtils
from jet.logger import logger

# Example usage
if __name__ == '__main__':
    print_utils = RedisPrintUtils()

    # Print all data from all databases
    logger.info("\nprint_all_data()")
    print_utils.print_all_data()

    # Print only 'key1' and 'key2' from DB 0
    logger.info("\nprint_data(0, ['key1', 'key2'])")
    print_utils.print_data(0, ['key1', 'key2'])

    # Print all keys from DB 1
    logger.info("\nprint_data(1)")
    print_utils.print_data(1)
