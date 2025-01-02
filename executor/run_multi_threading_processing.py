import time
from jet.executor import run_in_threads, run_in_processes, run_with_pool
from jet.logger import logger, time_it, sleep_countdown


@time_it
def test_time_it_annotation():
    time.sleep(3)


# For multiprocessing global scope

def test_time_it_named():
    time.sleep(3)


@time_it
def test_synchronous_calls():
    sleep_countdown(3, "Custom message")
    test_time_it_annotation()
    time_it(test_time_it_named)()


@time_it
def test_threaded_calls():
    tasks = [
        lambda: sleep_countdown(3, "Custom message"),
        test_time_it_annotation,
        test_time_it_named,
    ]
    run_in_threads(tasks)


# Global task wrapper for multi-processing calls
@time_it
def sample_multiprocessing_task():
    sleep_countdown(3, "Custom message")


@time_it
def test_multiprocessing_calls():
    tasks = [
        sample_multiprocessing_task,
        test_time_it_annotation,
        test_time_it_named,
    ]
    run_in_processes(tasks)


# Global task wrapper for multi-cpu pool global scope
@time_it
def sample_multi_cpu_task(message, duration):
    sleep_countdown(duration, message)
    return f"Completed {message} in {duration}s"


@time_it
def test_multi_cpu_pool_calls():
    tasks = [
        (sample_multi_cpu_task, "Task A", 3),
        (sample_multi_cpu_task, "Task B", 2),
        (sample_multi_cpu_task, "Task C", 1),
    ]

    # Use multiprocessing.Pool for parallel execution
    results = run_with_pool(tasks, pool_type="multiprocessing", processes=3)

    for result in results:
        logger.log(result, colors=["DEBUG", "SUCCESS"])


if __name__ == '__main__':
    logger.log("Running:", "test_threaded_calls", colors=["GRAY", "INFO"])
    test_threaded_calls()

    logger.log("Running:", "test_multiprocessing_calls",
               colors=["GRAY", "INFO"])
    test_multiprocessing_calls()

    logger.log("Running:", "test_multi_cpu_pool_calls",
               colors=["GRAY", "INFO"])
    test_multi_cpu_pool_calls()

    logger.log("Running:", "test_synchronous_calls", colors=["GRAY", "INFO"])
    test_synchronous_calls()
