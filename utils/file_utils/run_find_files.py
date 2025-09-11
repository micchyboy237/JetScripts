import os
import shutil
from jet.file.utils import save_file
from jet.utils.file_utils.search import find_files

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Test 1
args = {
    "base_dir": "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs",
    "include": ["en/"],
    "exclude": [],
    "include_content_patterns": [],
    "exclude_content_patterns": [],
    "case_sensitive": False,
    "extensions": [
        ".py",
        ".ipynb"
    ]
}
results = [
    file for file in find_files(**args)
]
expected = [
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/en/_config.py",
]
save_file({"args": args, "results": results, "expected": expected},
          f"{OUTPUT_DIR}/results1.json")

# Test 2
args = {
    "base_dir": "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs",
    "include": [],
    "exclude": [
        "zh/"
    ],
    "include_content_patterns": [],
    "exclude_content_patterns": [],
    "case_sensitive": False,
    "extensions": [
        ".py",
        ".ipynb"
    ]
}
results = [
    file for file in find_files(**args)
]
expected = [
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/en/_config.py",
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/hi/_config.py",
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/ko/_config.py",
]
save_file({"args": args, "results": results, "expected": expected},
          f"{OUTPUT_DIR}/results2.json")

# Test 3
args = {
    "base_dir": "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source",
    "include": [],
    "exclude": [
        "zh/"
    ],
    "include_content_patterns": [],
    "exclude_content_patterns": [],
    "case_sensitive": False,
    "extensions": []
}
results = [
    file for file in find_files(**args)
]
expected = [
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/en/_config.py",
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/hi/_config.py",
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/ko/_config.py",
]
save_file({"args": args, "results": results, "expected": expected},
          f"{OUTPUT_DIR}/results3.json")

# Test 4
args = {
    "base_dir": "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/en",
    "include": [],
    "exclude": [
        "zh/"
    ],
    "include_content_patterns": [],
    "exclude_content_patterns": [],
    "case_sensitive": False,
    "extensions": [
        ".ipynb"
    ]
}
results = [
    file for file in find_files(**args)
]
expected = [
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/en/_config.py",
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/hi/_config.py",
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/smolagents/docs/source/ko/_config.py",
]
save_file({"args": args, "results": results, "expected": expected},
          f"{OUTPUT_DIR}/results3.json")
