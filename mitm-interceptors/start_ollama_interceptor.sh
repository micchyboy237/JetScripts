#!/bin/bash

# Default to unset mode
MODE=""

CLIENT_LISTEN_PORT="11434"
CLIENT_TARGET_URL="http://jetairm1:11434"
CLIENT_PROXY_SCRIPT="ollama_proxy.py"

SERVER_LISTEN_PORT="11434"
SERVER_TARGET_URL="http://jetairm1:11435"
SERVER_PROXY_SCRIPT="ollama_interceptor.py"

# Parse command-line arguments
while getopts "cs" opt; do
  case $opt in
    c)
      MODE="client"
      ;;
    s)
      MODE="server"
      ;;
    *)
      echo "Usage: $0 -c (for client) or -s (for server)"
      exit 1
      ;;
  esac
done

# Ensure a mode is specified
if [ -z "$MODE" ]; then
  echo "Error: You must specify a mode. Use -c for client or -s for server."
  echo "Usage: $0 -c (for client) or -s (for server)"
  exit 1
fi

# Run mitmdump based on the selected mode
if [ "$MODE" == "client" ]; then
  echo "Running in client mode..."
  mitmdump --set validate_inbound_headers=false -s $CLIENT_PROXY_SCRIPT --mode reverse:$CLIENT_TARGET_URL -p $CLIENT_LISTEN_PORT
else
  echo "Running in server mode..."
  mitmdump --set validate_inbound_headers=false -s $SERVER_PROXY_SCRIPT --mode reverse:$SERVER_TARGET_URL -p $SERVER_LISTEN_PORT
fi

# Sample Commands
# Run server
# ./start_ollama_interceptor.sh -s
# Run client
# ./start_ollama_interceptor.sh -c