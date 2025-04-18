from rq import Worker
from redis import Redis

# Connect to Redis
redis_conn = Redis(host='localhost', port=6379, db=0)

# Get the worker instance
# Adjust to get the correct worker
worker = Worker.all(connection=redis_conn)[0]

# Print the list of failed jobs
job = worker.get_current_job()
print(f"Job: {job.id} | Status: {job.get_status()}")
