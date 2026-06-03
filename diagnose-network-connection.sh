# Check system-wide proxy settings
scutil --proxy

# Check network service proxies
networksetup -listallnetworkservices
networksetup -getwebproxy Wi-Fi
networksetup -getsecurewebproxy Wi-Fi
networksetup -getsocksfirewallproxy Wi-Fi

# Check firewall status
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# List apps with firewall rules
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --listapps

# Check stealth mode
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getstealthmode

# List active network interfaces
ifconfig | grep -E "^[a-z]"

# Check for VPN tunnels
netstat -rn | grep utun

# See all active connections
netstat -an | grep ESTABLISHED

# Check current DNS servers
scutil --dns | grep nameserver

# Check default route / gateway
netstat -rn | grep default

# Test connectivity step by step
ping -c 4 8.8.8.8     # Tests basic internet
ping -c 4 google.com  # Tests DNS resolution
traceroute google.com # Shows full network path

# Check for common third-party firewall processes
ps aux | grep -E "Little Snitch|LuLu|Hands Off|Murus"

# Check kernel extensions loaded
kmutil showloaded | grep -i "filter\|firewall\|vpn"

echo "=== PROXY ===" && scutil --proxy \
    && echo "=== FIREWALL ===" && sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate \
    && echo "=== DNS ===" && scutil --dns | grep nameserver | head -5 \
    && echo "=== ROUTES ===" && netstat -rn | grep default
