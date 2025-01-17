import unittest


class MyClass:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Name: {self.name}"

    def __repr__(self):
        return f"MyClass('{self.name}', {self.age})"

    def __format__(self, format_spec):
        if format_spec == "name":
            return f"Name: {self.name}"
        elif format_spec == "age":
            return f"Age: {self.age}"
        return f"MyClass object with name {self.name} and age {self.age}"


class TestMyClass(unittest.TestCase):
    def setUp(self):
        # Setting up a sample object for testing
        self.obj = MyClass("example", 30)

    def test_str_method(self):
        # Testing the __str__ method
        self.assertEqual(str(self.obj), "Name: example")

    def test_repr_method(self):
        # Testing the __repr__ method
        self.assertEqual(repr(self.obj), "MyClass('example', 30)")

    def test_format_method_name(self):
        # Testing the __format__ method with "name" format specifier
        self.assertEqual(f"{self.obj:name}", "Name: example")

    def test_format_method_age(self):
        # Testing the __format__ method with "age" format specifier
        self.assertEqual(f"{self.obj:age}", "Age: 30")

    def test_format_method_default(self):
        # Testing the __format__ method with a default format specifier
        self.assertEqual(
            f"{self.obj}", "MyClass object with name example and age 30")


if __name__ == "__main__":
    unittest.main()
