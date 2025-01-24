#!/bin/bash

# Default client configuration
CLIENT_LISTEN_PORT="11434"
CLIENT_TARGET_URL="http://jetairm1:11434"
CLIENT_PROXY_SCRIPT="ollama_proxy.py"

# Run mitmdump in client mode
echo "Running in client mode..."
mitmdump -s $CLIENT_PROXY_SCRIPT --mode reverse:$CLIENT_TARGET_URL -p $CLIENT_LISTEN_PORT

# Sample Command
# Run client
# ./run_proxy_sample.sh