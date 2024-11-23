#!/bin/bash

# Default to server mode
MODE="server"
PORT="11434"
TARGET_URL="http://jetairm1:11435"
SCRIPT="ollama_interceptor.py"

# Parse command-line arguments
while getopts "c" opt; do
  case $opt in
    c)
      MODE="client"
      TARGET_URL="http://jetairm1:11434"
      ;;
    *)
      echo "Usage: $0 [-c for client]"
      exit 1
      ;;
  esac
done

# Run mitmdump based on the selected mode
if [ "$MODE" == "client" ]; then
  echo "Running in client mode..."
  mitmdump --mode reverse:$TARGET_URL -p $PORT
else
  echo "Running in server mode..."
  mitmdump -s $SCRIPT --mode reverse:$TARGET_URL -p $PORT
fi
