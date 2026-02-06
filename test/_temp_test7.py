from jet.code.python_code_extractor import move_imports_to_top

python_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.py"

with open(python_file, encoding="utf-8") as f:
    source_code = f.read()

updated_code = move_imports_to_top(source_code)
print(f"Code:\n{updated_code}")
