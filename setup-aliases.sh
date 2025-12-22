# Set keyboard shortcut aliases / functions
# alias freeze="pipdeptree -l -d 0 --python .venv/bin/python --freeze | grep -E '^\S' > requirements-frozen.txt"

deps() {
    local package=""
    local verbose=0

    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p) package="$2"; shift 2;;
            -v) verbose=1; shift;;
            *) package="$1"; shift;;
        esac
    done

    # Ensure package is provided
    if [[ -z "$package" ]]; then
        echo -e "Error: Package must be provided\n\nUsage:\n  deps <package>\n  deps -p <package> [-v]\n\nExample:\n  deps numpy\n  deps -p numpy -v"
        return 1
    fi

    # Fetch package details, dependents, and dependencies using Python
    python3 -c "
import importlib.util
import pkg_resources

def get_package_location(pkg):
    spec = importlib.util.find_spec(pkg)
    return spec.origin if spec else None

def print_dependencies(title, items):
    if items:
        print(title)
        for name, version, location in items:
            print(f'  - {name} ({version})' + (f': {location}' if $verbose and location else ''))
    else:
        print(f'{title} None')

try:
    target = '$package'
    verbose = bool($verbose)
    
    # Main package details
    main_dist = pkg_resources.get_distribution(target)
    main_version = main_dist.version
    main_location = get_package_location(target) if verbose else None
    print(f'Package: {target} ({main_version})')
    if verbose and main_location:
        print(f'Location: {main_location}')
    print('')
    
    # Dependents (packages that require this package)
    dependents = [(d.project_name, d.version, get_package_location(d.project_name) if verbose else None)
                  for d in pkg_resources.working_set if target.lower() in [r.project_name.lower() for r in d.requires()]]
    print_dependencies('Packages that require ' + target + ':', dependents)
    print('')
    
    # Dependencies (packages required by this package)
    dependencies = [(r.project_name, pkg_resources.get_distribution(r.project_name).version,
                     get_package_location(r.project_name) if verbose else None)
                    for r in main_dist.requires()]
    print_dependencies('Dependencies:', dependencies)
    
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
}

deps_tree() {
    # Default path to the Python executable inside a virtual environment
    local python_path=".venv/bin/python"
    # Default output file for dependency tree
    local output_path="requirements-deps-tree.json"
    # Default Python path if the specified one is unavailable
    local default_python="$HOME/.pyenv/shims/python"
    # Default package (empty means show all dependencies)
    local package=""

    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
    case $1 in
        -f)  # Custom Python interpreter path
        python_path="$2"
        shift 2
        ;;
        -o)  # Custom output file path (not currently used)
        output_path="$2"
        shift 2
        ;;
        -p)  # Specific package for dependency tree
        package="$2"
        shift 2
        ;;
        *)  # Handle unknown options
        echo "Unknown option: $1"
        return 1
        ;;
    esac
    done

    # Check if the specified Python executable exists and is executable
    if [[ ! -x "$python_path" ]]; then
        echo "Specified Python path '$python_path' does not exist or is not executable. Falling back to '$default_python'."
        python_path="$default_python"
    fi

    # Check if the default Python executable exists and is executable
    if [[ ! -x "$python_path" ]]; then
        echo "Error: Default Python path '$default_python' does not exist or is not executable."
        return 1
    fi

    # Check if pipdeptree is installed
    if ! "$python_path" -m pipdeptree --version &>/dev/null; then
        echo "Error: pipdeptree is not installed. Install it using:"
        echo "  pip install pipdeptree"
        return 1
    fi

    # Construct the pipdeptree command
    local command="pipdeptree -l -d 0 --python \"$python_path\" --json-tree"

    # If a package is specified, filter for that package
    if [[ -n "$package" ]]; then
        # Check if jq is installed
        if ! command -v jq &>/dev/null; then
            echo "Error: 'jq' is required for filtering but is not installed."
            echo "  Install it using: brew install jq (macOS) or sudo apt-get install jq (Linux)"
            return 1
        fi

        command="$command | jq '.[] | select(.package.name == \"$package\")'"
    fi

    echo "$command"  # Display the command being executed
    eval "$command" || echo "Error: Command failed. Ensure pipdeptree is installed and working."

    # -------------------------------
    # How to Use:
    # 1. Run the function with the default Python interpreter:
    #    deps
    #    - Uses `.venv/bin/python` to generate a full dependency tree.
    #
    # 2. Specify a different Python interpreter:
    #    deps -f /path/to/python
    #    - Allows you to specify a custom Python executable.
    #
    # 3. Get dependencies of a specific package:
    #    deps -p package_name
    #    - Filters the dependency tree to show only dependencies of `package_name`.
    #
    # 4. Specify an output file (not currently used but parsed):
    #    deps -o output.json
    #    - Parses the `-o` flag, but output saving is not yet implemented.
    #
    # Note: This function relies on `pipdeptree` and `jq`. Install them if not already installed:
    #       pip install pipdeptree
    #       brew install jq  # (or use sudo apt-get install jq on Linux)
}

freeze() {
    # Usage examples:
    # 1. Run with default settings (uses .venv/bin/python and outputs to requirements-frozen.txt):
    #    freeze
    # 2. Specify a custom Python path:
    #    freeze -f /usr/bin/python3
    # 3. Specify a custom output file:
    #    freeze -o requirements.txt
    # 4. Specify both custom Python path and output file:
    #    freeze -f /usr/bin/python3 -o custom-requirements.txt
    # 5. Invalid option (will print error):
    #    freeze -x (returns "Unknown option: -x")
    # 6. Non-executable Python path (falls back to default_python):
    #    freeze -f /invalid/path/python
    # 7. Non-executable default Python path (returns error):
    #    freeze -f /invalid/path/python (if default_python is also invalid)

    local python_path=".venv/bin/python"
    local output_path="requirements-frozen.txt"
    local default_python="$HOME/Desktop/External_Projects/Jet_Projects/jet_python_modules/.venv/bin/python"

    while [[ $# -gt 0 ]]; do
    case $1 in
        -f)
        python_path="$2"
        shift 2
        ;;
        -o)
        output_path="$2"
        shift 2
        ;;
        *)
        echo "Unknown option: $1"
        return 1
        ;;
    esac
    done

    if [[ ! -x "$python_path" ]]; then
    echo "Specified Python path '$python_path' does not exist or is not executable. Falling back to '$default_python'."
    python_path="$default_python"
    fi

    if [[ ! -x "$python_path" ]]; then
    echo "Error: Default Python path '$default_python' does not exist or is not executable."
    return 1
    fi

    # Prepend the Python path as a comment to the output file
    echo "# $python_path" > "$output_path"

    # Generate the requirements and append to the file
    local command="pipdeptree -l -d 0 --python \"$python_path\" --freeze | grep -E '^\S' >> \"$output_path\""
    echo "$command"
    eval "$command"
}

size() {
    local base_dir="."
    
    while [[ $# -gt 0 ]]; do
        case $1 in
        -b)
            base_dir="$2"
            shift 2
            ;;
        *)
            base_dir="$1"
            shift
            ;;
        esac
    done

    # local command="find $base_dir -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 du -sh | sort -h"
    # Command to get the sizes of subdirectories and total size
    # local command="du -sh $base_dir && find $base_dir -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 du -sh | sort -h"
    local command="find $base_dir -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 du -sh | sort -h; du -sh $base_dir"
    echo "$command"
    eval "$command"
}

activate_venv() {
    if [[ -f .venv/bin/activate ]]; then
        source .venv/bin/activate
    # else
    #     echo "No virtual environment found in the current directory."
    fi
}


deactivate_venv() {
    deactivate
    rm -rf .venv
}

setup_venv() {
    deactivate_venv
    python -m venv .venv
    activate_venv

    echo "which pip"
    which pip
    echo ""

    echo "which python"
    which python
    echo ""

    echo "pip --version"
    pip --version
    echo ""
    
    echo "python --version"
    python --version
    echo ""
}


freeze_venv() {
    freeze -f .venv/bin/python -o requirements-frozen.txt
}

reinstall_venv() {
    setup_venv
    python --version
    pip install -r requirements.txt
}

force_reinstall_venv() {
    activate_venv
    pip install --force-reinstall
}

reset_venv() {
    local current_dir=$PWD

    source ~/.zshrc
    cd ~
    cd "$current_dir"
}

reinstall_python() {
    pyenv uninstall 3.12.7
    pyenv install 3.12.7
    pip install -r ~/requirements.txt
    python --version
}

# pip() {
#     activate_venv

#     command python -m pip "$@"
# }

# cd() {
#     echo "Arguments: $@"
#     command cd "$@" && activate_venv
# }

large_folders() {
    local base_dir=""
    local min_size=100
    local includes=""
    local excludes=""
    local max_depth=""
    local limit=""
    local output_file=""
    local max_backward_depth=""
    local delete=false
    local direction=""
    local save=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -b) base_dir="$2"; shift 2;;
            -s) min_size="$2"; shift 2;;
            -i) includes="$2"; shift 2;;
            -e) excludes="$2"; shift 2;;
            -d) max_depth="$2"; shift 2;;
            -l) limit="$2"; shift 2;;
            -f) output_file="$2"; shift 2;;
            --max-backward-depth) max_backward_depth="$2"; shift 2;;
            --delete) delete=true; shift;;
            --direction) direction="$2"; shift 2;;
            --save) save=true; shift;;  # Add condition for --save
            *) shift;;
        esac
    done

    local args=()
    [[ -n "$base_dir" ]] && args+=("-b" "$base_dir")
    [[ -n "$min_size" ]] && args+=("-s" "$min_size")
    [[ -n "$includes" ]] && args+=("-i" "$includes")
    [[ -n "$excludes" ]] && args+=("-e" "$excludes")
    [[ -n "$max_depth" ]] && args+=("-d" "$max_depth")
    [[ -n "$limit" ]] && args+=("-l" "$limit")
    [[ -n "$output_file" ]] && args+=("-f" "$output_file")
    [[ -n "$max_backward_depth" ]] && args+=(--max-backward-depth "$max_backward_depth")
    [[ "$delete" == true ]] && args+=(--delete)
    [[ -n "$direction" ]] && args+=(--direction "$direction")
    [[ "$save" == true ]] && args+=(--save)  # Add --save to arguments if specified

    python /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/find_large_folders.py "${args[@]}"

    # Example:
    # large_folders -b ~/Desktop/External_Projects -s 50 --save
    # large_folders -b . -s 100 -i "**/*" -e "node_modules,.venv" -d 2 -l 10 -f "out.json" --delete --direction forward --max-backward-depth 3 --save
    # large_folders -b ~/Desktop/External_Projects -i "node_modules,.venv,venv" -d 0 --save
    # large_folders -b /Users/jethroestrada/.cache/huggingface/hub -s 200 -d 0 --save
}

last_updates() {
    local base_dir=""
    local extensions=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -b) base_dir="$2"; shift 2;;
            -e) extensions="$2"; shift 2;;
            *) shift;;
        esac
    done

    local args=()
    [[ -n "$base_dir" ]] && args+=("$base_dir")
    [[ -n "$extensions" ]] && args+=("-e" "$extensions")

    python /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_notes/python_scripts/git_stats.py "${args[@]}"

    # Example:
    # last_updates -b . -e ".py,.md"
}

git_stats() {
    local base_dir="."
    local extensions=""
    local depth=""
    local mode=""
    local type_filter=""
    local file_pattern=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e) extensions="$2"; shift 2;;
            -d) depth="$2"; shift 2;;
            -m) mode="$2"; shift 2;;
            -t) type_filter="$2"; shift 2;;
            -p) file_pattern="$2"; shift 2;;
            *) base_dir="$1"; shift;;
        esac
    done

    local args=()
    args+=("$base_dir")
    [[ -n "$extensions" ]] && args+=("-e" "$extensions")
    [[ -n "$depth" ]] && args+=("-d" "$depth")
    [[ -n "$mode" ]] && args+=("-m" "$mode")
    [[ -n "$type_filter" ]] && args+=("-t" "$type_filter")
    [[ -n "$file_pattern" ]] && args+=("-p" "$file_pattern")

    python /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_notes/python_scripts/git_stats.py "${args[@]}"

    # Examples for dirs mode:
    # git_stats . -d 1 -t dirs

    # General Examples
    # git_stats
    # git_stats -e ".py,.ipynb"
    # git_stats -e ".py" -p "test_*.py"
    # git_stats -e ".py,.md" -t files -m file
    # git_stats -e ".py,.md" -t dirs -m git -d 3
    # git_stats -e ".py,.ipynb" -m auto
    # git_stats -p "*mcp*,*MCP*" -e ".py,.ipynb,.mdx"
}

# Function to check memory usage of specific Python processes
mem_python_check() {
  local threshold="${1:-0}"
  local python_version="python3.12"
  local mem_column=8

  top -l 1 -o mem | grep "$python_version" | awk -v th="$threshold" -v col="$mem_column" '$col ~ /[0-9]+M/ && $col+0 > th'
}

# Usage:
#   mem_python_check            # Lists python3.12 processes using more than 0 MB (default)
#   mem_python_check 300        # Lists python3.12 processes using more than 300 MB


# Function to kill specific Python processes exceeding memory threshold
mem_python_kill() {
  local threshold="${1:-0}"
  local python_version="python3.12"
  local mem_column=8

  top -l 1 -o mem | grep "$python_version" | awk -v th="$threshold" -v col="$mem_column" '$col ~ /[0-9]+M/ && $col+0 > th {print $1}' | xargs kill -9
}

# Usage:
#   mem_python_kill             # Kills python3.12 processes using more than 0 MB (default)
#   mem_python_kill 300         # Kills python3.12 processes using more than 300 MB


# Override pytest
pytest() {
  if [[ "$*" == *--watch* ]]; then
    # Remove --watch from arguments to pass to pytest-watcher
    local args=("${@/--watch/}")
    command python -m pytest_watcher . -- --last-failed -vv "${args[@]}"
  else
    command python -m pytest -vv --last-failed "$@"
  fi
}

# Usage:
#   pytest                    # Run all tests with verbose output and last-failed mode
#   pytest test_file.py       # Run specific test file
#   pytest -k "test_name"     # Run tests matching the given name
#   pytest --watch            # Run tests in watch mode with pytest-watcher
#   pytest --watch test_file.py  # Watch specific test file


# Check if the 'deps' function is already defined to prevent echo
# if ! declare -f deps &>/dev/null; then
echo "Added deps, deps_tree, freeze, size, setup_venv, freeze_venv, reinstall_venv, force_reinstall_venv, activate_venv, deactivate_venv, reinstall_python functions, pip, large_folders, last_updates, git_stats, mem_python_check, mem_python_kill, pytest"
# fi
