from jet.wordnet.utils import sliding_window

if __name__ == "__main__":
    # Example usage on a list of numbers
    numbers = [1, 2, 3, 4, 5, 6, 7]
    window_size = 2
    step_size = 1

    print("Sliding windows on a list of numbers:")
    for sequence in sliding_window(numbers, window_size, step_size):
        print(sequence)