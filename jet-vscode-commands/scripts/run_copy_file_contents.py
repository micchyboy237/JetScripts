import sys
import pyperclip  # Install this with `pip install pyperclip`


def main():
    if len(sys.argv) != 2:
        print("Usage: python copy_file_contents.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, 'r') as file:
            content = file.read()
            pyperclip.copy(content)
            print(f"Contents of {file_path} copied to clipboard!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
