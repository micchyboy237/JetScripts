
from typing import TypedDict


class PersonTypedDict(dict):
    name: str
    age: int


person_typed_dict = PersonTypedDict(name="Alice", age=25)
print(person_typed_dict.__class__.__name__)  # Output: PersonTypedDict


class PersonTypedDict(TypedDict):
    name: str
    age: int


person_typed_dict = PersonTypedDict(name="Alice", age=25)
print(person_typed_dict.__class__.__name__)  # Output: dict
