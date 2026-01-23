from jet.wordnet.utils import increasing_window


if __name__ == "__main__":
    # Example usage on a list of numbers
    numbers = [1, 2, 3, 4, 5, 6, 7]
    step_size = 1    # Move the window by one token each time

    # Generate and print the sequences
    results = increasing_window(numbers, step_size)

    # Print only those with more than 1 item
    filtered_result = [seq for seq in results if len(seq) > 1]
    for sequence in filtered_result:
        print(sequence)
