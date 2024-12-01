from jet.cache.redis import RedisUseCases

# Example Usage:
if __name__ == '__main__':
    redis_client = RedisUseCases()

    # Call the method to cache the product details
    print(redis_client.get_product_details(1))

    # Check the cached value directly
    cached_value = redis_client.redis.get("product:1")
    print(f"Cached value for product:1 is: {cached_value.decode('utf-8')}")
