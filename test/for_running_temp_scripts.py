import time
from jet.logger import logger
from jet.logger.timer import SleepInterruptible, SleepStatus


# Example Usage
def on_sleep_complete(status: SleepStatus, total_elapsed: float, restart_elapsed: float) -> None:
    print(f"Sleep Status: {status} | Slept: {
          total_elapsed:.2f} seconds | Restarted: {restart_elapsed:.2f}")


def main_cancel_sample():
    sleep_duration = 3  # Duration to sleep
    sleep_manager = SleepInterruptible(
        sleep_duration, on_complete=on_sleep_complete)

    logger.newline()
    logger.info("Cancel Example...")
    print("Starting sleep...")
    sleep_manager.start_sleep()

    time.sleep(2)  # Let it sleep for 2 seconds

    # Check status before canceling
    print(f"Sleep status before cancel: {sleep_manager.get_status()}")
    print("Cancelling sleep...")
    sleep_manager.cancel_sleep()

    # Check status after canceling
    print(f"Sleep status after cancel: {sleep_manager.get_status()}")


def main_restart_sample():
    sleep_duration = 3  # Duration to sleep
    sleep_manager = SleepInterruptible(
        sleep_duration, on_complete=on_sleep_complete)

    # Restart the sleep
    logger.newline()
    logger.info("Restart Example...")
    print("Starting sleep...")
    sleep_manager.start_sleep()

    time.sleep(2)  # Let it sleep again for 2 seconds

    print("Restarting sleep...")
    sleep_manager.restart_sleep()

    # Check status after restarting
    print(f"Sleep status after restart: {sleep_manager.get_status()}")


def main_wait_over_duration_sample():
    sleep_duration = 3  # Duration to sleep
    sleep_manager = SleepInterruptible(
        sleep_duration, on_complete=on_sleep_complete)

    # Start again and let it complete
    logger.newline()
    logger.info("Wait > Duration  Example...")
    print("Starting sleep again...")
    sleep_manager.start_sleep()
    time.sleep(5)  # Main thread sleeps longer
    print(f"Sleep status after completion: {sleep_manager.get_status()}")


if __name__ == "__main__":
    main_cancel_sample()
    main_restart_sample()
    main_wait_over_duration_sample()
