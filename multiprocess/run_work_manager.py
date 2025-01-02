import sys
import threading
import time
from jet.logger import logger, asleep_countdown
from jet.multiprocess import WorkManager


# Example usage
if __name__ == '__main__':
    def example_work(task_id):
        print(f"Executing task {task_id}")

    # Define work callbacks
    work_callbacks = [lambda: example_work(i) for i in range(1, 101)]

    work_manager = WorkManager(work_callbacks=work_callbacks, num_threads=2)
    work_manager.start_threads()

    work_manager.pause_thread(1, timeout=5, delay=3)
    work_manager.pause_thread(2, timeout=8, delay=10)
