from pathlib import Path

folder = Path("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/translators")

for file in folder.iterdir():
    if file.is_file():
        new_name = file.with_name(f"translate_{file.name}")
        file.rename(new_name)
