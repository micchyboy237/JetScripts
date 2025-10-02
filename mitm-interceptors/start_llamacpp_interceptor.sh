# TODO: Check why going through this interceptor is slow

# CLIENT_LISTEN_PORT="8080"
# CLIENT_TARGET_URL="http://shawn-pc.local:8080"
# CLIENT_PROXY_SCRIPT="llamacpp_interceptor.py"

# echo "Running client proxy..."
# mitmdump -s $CLIENT_PROXY_SCRIPT --mode reverse:$CLIENT_TARGET_URL -p $CLIENT_LISTEN_PORT
