from typing import List
METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False


def run_tests(candidate):
    try:
        check(candidate)
        print("ALL TESTS PASSED !")
        print("TERMINATE")
    except AssertionError as e:
        print(f"SOME TESTS FAILED - TRY AGAIN! Error: {str(e)}")


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    Check if in given list of numbers, are any two numbers closer to each other than 
    given threshold.

    Args:
        numbers (List[float]): A list of floating point numbers.
        threshold (float): The minimum difference required between two numbers.

    Returns:
        bool: True if any two numbers are closer than the threshold, False otherwise.
    """

    # Sort the list in ascending order
    numbers.sort()

    # Iterate over the sorted list to compare each number with its next one
    for i in range(len(numbers) - 1):

        # Calculate the difference between the current and next number
        diff = abs(numbers[i] - numbers[i + 1])

        # If the difference is less than or equal to the threshold, return True
        if diff <= threshold:
            return True

    # If no pair of numbers with a difference less than or equal to the threshold is found, return False
    return False


run_tests(has_close_elements)
