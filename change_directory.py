import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python change_directory.py <path>")
        return
    path = sys.argv[1]
    print(path)


if __name__ == "__main__":
    main()

# Sample command
# eval cd $(python /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/change_directory.py /Users/jethroestrada/Desktop/External_Projects/AI/eval_agents/helicone)
