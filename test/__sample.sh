# Native zsh equivalent to: dirname $(realpath $0)
SCRIPT_DIR="${0:A:h}"
SCRIPT_STEM="${0:A:t:r}"
OUTPUT_DIR="${SCRIPT_DIR}/generated/${SCRIPT_STEM}"

echo "Output directory is: $OUTPUT_DIR"