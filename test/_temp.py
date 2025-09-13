from typing import List
METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == [
        '(()())', '((()))', '()', '((())()())'
    ]
    assert candidate('() (()) ((())) (((())))') == [
        '()', '(())', '((()))', '(((())))'
    ]
    assert candidate('(()(())((())))') == [
        '(()(())((())))'
    ]
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']


def run_tests(candidate):
    try:
        check(candidate)
        print("ALL TESTS PASSED !")
        print("TERMINATE")
    except AssertionError:
        print("SOME TESTS FAILED - TRY AGAIN !")


def separate_paren_groups(paren_string: str) -> List[str]:
    """
    Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those group into separate strings and return the list of those.

    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.

    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    # Initialize an empty stack to keep track of opening parentheses
    stack = []

    # Initialize an empty list to store the result
    result = []

    # Iterate over each character in the input string
    for char in paren_string:
        # If the character is an opening parenthesis, push it onto the stack
        if char == '(':
            stack.append(char)
        # If the character is a closing parenthesis
        elif char == ')':
            # If the stack is not empty, pop the corresponding opening parenthesis from the stack
            if stack:
                stack.pop()
            # If the stack is empty, it means we have found a complete group of parentheses
            else:
                # Join all characters in the result list into a single string and add it to the result list
                result.append(''.join(result))
                # Reset the result list for the next group
                result = []

    # If there are any remaining opening parentheses in the stack, it means they do not have a corresponding closing parenthesis
    if stack:
        raise ValueError("Unbalanced parentheses")

    return result


run_tests(separate_paren_groups)
