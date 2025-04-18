from redis import Redis
from rq import Queue
from rq.registry import FailedJobRegistry

# Connect to Redis (default localhost and port)
redis_conn = Redis()
queue = Queue('default', connection=redis_conn)

# Access the failed job registry
failed_registry = FailedJobRegistry(queue=queue)

# Get all failed job IDs
failed_job_ids = failed_registry.get_job_ids()

for job_id in failed_job_ids:
    job = queue.fetch_job(job_id)
    print(f"\n--- Job ID: {job.id} ---")
    print(f"Function: {job.func_name}")
    print(f"Args: {job.args}")
    print(f"Kwargs: {job.kwargs}")
    print(f"Enqueued at: {job.enqueued_at}")
    print(f"Failed at: {job.ended_at}")
    print("Exception:\n", job.exc_info)
