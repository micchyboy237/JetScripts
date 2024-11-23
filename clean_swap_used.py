import os
import subprocess
import time


def convert_size_to_mb(size: int):
    return size / 1000  # Convert to MB


def format_size(size_mb):
    if size_mb >= 1000:
        return f"{size_mb / 1000:.2f} GB"
    return f"{size_mb:.2f} MB"


def check_swap_usage():
    """Check current swap usage by using 'vm_stat' command."""
    try:
        # Run vm_stat to get the swap usage
        result = subprocess.run(['vm_stat'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            swap_usage = 0
            for line in lines:
                if "swapins" in line.lower() or "swapouts" in line.lower():
                    # Extract the number and remove any non-numeric characters
                    value = line.split(":")[1].strip()
                    # Keep only digits
                    value = ''.join(filter(str.isdigit, value))
                    if value:
                        swap_usage += int(value)
            return convert_size_to_mb(swap_usage)
        else:
            print("Error retrieving vm_stat output.")
            return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def free_swap_space():
    """Try to free swap space on MacOS by clearing inactive memory."""
    try:
        subprocess.check_call("sudo purge", shell=True)
        print("Swap space cleared.")
    except subprocess.CalledProcessError:
        print("Error occurred while clearing swap space.")


if __name__ == "__main__":
    # Check initial swap usage
    print("Checking initial swap usage...")
    swap_usage_before = check_swap_usage()

    if swap_usage_before is not None:
        print(f"Initial swap usage: {format_size(swap_usage_before)}")

        # Free swap space if it exceeds a threshold (example: 1 GB or more)
        if swap_usage_before > 1000:  # Threshold in MB
            free_swap_space()
        else:
            print("Swap usage is within acceptable limits.")
    else:
        print("Error retrieving swap usage.")

    swap_usage_after = check_swap_usage()
    if swap_usage_after is not None:
        print(f"Swap usage after freeing: {format_size(swap_usage_after)}")
    else:
        print("Failed to check swap usage.")
