# Sample Python file with unused code

def used_function():
    print("This function is used.")


def unused_function():
    print("This function is not used.")


class UnusedClass:
    def __init__(self):
        print("This class is not used.")


class UsedClass:
    def __init__(self):
        print("This class is used.")

    def used_method(self):
        print("This method is used.")


# Example usage
if __name__ == "__main__":
    used_function()
    instance = UsedClass()
    instance.used_method()
