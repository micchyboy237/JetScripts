============================
Sample reStructuredText File
============================

Introduction
============
This file contains various Python code blocks, demonstrating different structures and directives.

Basic Python Code Block
------------------------
Here is a simple Python code block:

.. code-block:: python

    def hello_world():
        print("Hello, World!")

Another Block
-------------
Another example of a Python function:

.. code-block:: python

    def add_numbers(a, b):
        return a + b

Inline Python
-------------
Inline code is typically not part of code blocks but looks like this: ``print("Hello, Inline!")``.
This should not be extracted as a code block.

Block Example
-------------
A Python block without a title (caption removed):

.. code-block:: python

    class Greeter:
        def __init__(self, name):
            self.name = name
        
        def greet(self):
            return f"Hello, {self.name}!"

Multiline Strings
-----------------
Code blocks can also contain multiline strings:

.. code-block:: python

    long_text = """
    This is a multiline string.
    It spans multiple lines.
    """

Non-Python Code
---------------
This block is not Python and should not be extracted:

.. code-block:: javascript

    function helloWorld() {
        console.log("Hello, JavaScript!");
    }

Indented Code
-------------
Indented Python block:

.. code-block:: python

        if __name__ == "__main__":
            print("This is an indented block.")
