import os
import shutil
from jet.file.utils import save_file
from jet.utils.file_utils.get_folders import get_folder_absolute_paths

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == "__main__":
    directory_path = "/Users/jethroestrada/Desktop/External_Projects/AI"
    depth = 2

    folder_paths = get_folder_absolute_paths(directory_path, depth)
    for path in folder_paths:
        print(path)

    base_folder_name = os.path.splitext(os.path.basename(directory_path))[0]
    save_file({
        "dir": directory_path,
        "count": len(folder_paths),
        "folders": folder_paths,
    }, f"{OUTPUT_DIR}/{base_folder_name}_folders.json")
