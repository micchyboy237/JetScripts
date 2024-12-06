# Set keyboard shortcut aliases / functions
# alias freeze="pipdeptree -l -d 0 --python .venv/bin/python --freeze | grep -E '^\S' > requirements.txt"
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
  pipdeptree -l -d 0 --python "$python_path" --freeze | grep -E '^\S' >> "$output_path"
}
