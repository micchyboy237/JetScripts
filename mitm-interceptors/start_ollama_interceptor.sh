#!/bin/bash

# Default to unset mode
MODE=""

# Client 1 configuration
CLIENT1_LISTEN_PORT="11434"
# CLIENT1_TARGET_URL="http://jethros-macbook-air.local:11434"
CLIENT1_TARGET_URL="http://shawn-pc.local:11434"
CLIENT1_PROXY_SCRIPT="ollama_proxy.py"

# Client 2 configuration
CLIENT2_LISTEN_PORT="11435"
CLIENT2_TARGET_URL="http://shawn-pc.local:11434"
CLIENT2_PROXY_SCRIPT="ollama_proxy.py"

# Server configuration
SERVER_LISTEN_PORT="11434"
SERVER_TARGET_URL="http://jethros-macbook-air.local:11435"
SERVER_PROXY_SCRIPT="ollama_reverse_proxy.py"

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
      echo "Usage: $0 -c (for clients) or -s (for server)"
      exit 1
      ;;
  esac
done

# Ensure a mode is specified
if [ -z "$MODE" ]; then
  echo "Error: You must specify a mode. Use -c for clients or -s for server."
  echo "Usage: $0 -c (for clients) or -s (for server)"
  exit 1
fi

# Run mitmdump based on the selected mode
if [ "$MODE" == "client" ]; then
  echo "Running in client mode..."
  echo "Starting client 1 (target: $CLIENT1_TARGET_URL)..."
  mitmdump --set validate_inbound_headers=false -s $CLIENT1_PROXY_SCRIPT --mode reverse:$CLIENT1_TARGET_URL -p $CLIENT1_LISTEN_PORT &
  echo "Starting client 2 (target: $CLIENT2_TARGET_URL)..."
  mitmdump --set validate_inbound_headers=false -s $CLIENT2_PROXY_SCRIPT --mode reverse:$CLIENT2_TARGET_URL -p $CLIENT2_LISTEN_PORT &
  wait
else
  echo "Running in server mode (target: $SERVER_TARGET_URL)..."
  mitmdump --set validate_inbound_headers=false -s $SERVER_PROXY_SCRIPT --mode reverse:$SERVER_TARGET_URL -p $SERVER_LISTEN_PORT
fi

# Sample Commands
# Run server
# ./start_ollama_interceptor.sh -s
# Run both clients
# ./start_ollama_interceptor.sh -c