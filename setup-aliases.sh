# Set keyboard shortcut aliases / functions
# alias freeze="pipdeptree -l -d 0 --python .venv/bin/python --freeze | grep -E '^\S' > requirements.txt"
deps() {
    local python_path=".venv/bin/python"
    local output_path="requirements.txt"
    local default_python="$HOME/.pyenv/shims/python"

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

    local command="pipdeptree -l -d 0 --python \"$python_path\" --json-tree"
    echo "$command"
    eval "$command"
}

freeze() {
    local python_path=".venv/bin/python"
    local output_path="requirements.txt"
    local default_python="$HOME/.pyenv/shims/python"

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

sizes() {
    local base_dir="/Users/jethroestrada/Desktop/External_Projects"
    local min_size=0
    local include="<folder>/bin/activate"

    while [[ $# -gt 0 ]]; do
        case $1 in
        -b)
            base_dir="$2"
            shift 2
            ;;
        -s)
            min_size="$2"
            shift 2
            ;;
        -i)
            include="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            return 1
            ;;
        esac
    done

    python /Users/jethroestrada/Desktop/External_Projects/JetScripts/find_large_folders.py -b "$base_dir"  -s $min_size -i "$include"
}

activate_venv() {
    local current_dir=$PWD
    local activation_script="$current_dir/.venv/bin/activate"

    if [ -f "$activation_script" ]; then
        source "$activation_script"
    # else
    #     echo "Activation script not found: $activation_script"
    fi
}

deactivate_venv() {
    deactivate
    rm -rf .venv
}

setup_venv() {
    deactivate_venv
    python -m venv .venv
    source .venv/bin/activate
    which pip
    which python
}


freeze_venv() {
    freeze -f .venv/bin/python -o requirements.txt
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

reinstall_python() {
    pyenv uninstall 3.12.7
    pyenv install 3.12.7
    pip install -r ~/requirements.txt
    python --version
}

pip() {
    activate_venv

    # # Check if the command is 'pip uninstall'
    # if [ "$1" = "uninstall" ]; then
    #     # Ensure pip-autoremove is installed
    #     if ! command -v pip-autoremove &>/dev/null; then
    #         echo "pip-autoremove is not installed. Installing it now..."
    #         command pip install pip-autoremove
    #     fi

    #     # Use pip-autoremove for uninstalling
    #     echo "Using pip-autoremove to uninstall $2 and its dependencies..."
    #     pip-autoremove "$2" -y
    # else
    #     # Pass other commands to the original pip
    #     command pip "$@"
    # fi

    command pip "$@"
}

# Check if the 'deps' function is already defined to prevent echo
# if ! declare -f deps &>/dev/null; then
echo "Added deps, freeze, size, sizes, setup_venv, freeze_venv, reinstall_venv, force_reinstall_venv, activate_venv, deactivate_venv, reinstall_python functions, pip"
# fi
