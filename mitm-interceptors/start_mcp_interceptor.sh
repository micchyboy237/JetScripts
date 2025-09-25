#!/bin/zsh

CLIENT_LISTEN_PORT="8931"
CLIENT_TARGET_URL="http://localhost:8932"
CLIENT_PROXY_SCRIPT="mcp_interceptor.py"

# Run mitmdump in client mode
echo "Running client proxy..."
mitmdump -s $CLIENT_PROXY_SCRIPT --mode reverse:$CLIENT_TARGET_URL -p $CLIENT_LISTEN_PORT
