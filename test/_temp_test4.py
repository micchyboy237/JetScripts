from jet.utils.code_utils import normalize_multiline_spaces

text = """
    Release Date
    : January 1, 2025
    
    Platforms
    : PC
    : Mac


    Genre
    : Comedy
"""
expected = """
Release Date
: January 1, 2025

Platforms
: PC
: Mac

Genre
: Comedy
""".strip()



result = normalize_multiline_spaces(text)
print("Result:")
print(result)
assert result == expected