#!/bin/bash

# Default to server mode
MODE="server"

CLIENT_LISTEN_PORT="11434"
CLIENT_TARGET_URL="http://jetairm1:11434"
CLIENT_PROXY_SCRIPT="ollama_proxy.py"

SERVER_LISTEN_PORT="11434"
SERVER_TARGET_URL="http://jetairm1:11435"
SERVER_PROXY_SCRIPT="ollama_interceptor.py"

# Parse command-line arguments
while getopts "c" opt; do
  case $opt in
    c)
      MODE="client"
      SERVER_TARGET_URL="http://jetairm1:11434"
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
  mitmdump -s $CLIENT_PROXY_SCRIPT --mode reverse:$SERVER_TARGET_URL -p $CLIENT_LISTEN_PORT
else
  echo "Running in server mode..."
  mitmdump -s $SERVER_PROXY_SCRIPT --mode reverse:$SERVER_TARGET_URL -p $SERVER_LISTEN_PORT
fi

# Sample Commands
# Run server
# ./start_ollama_interceptor.sh
# Run client
# ./start_ollama_interceptor.sh -c
