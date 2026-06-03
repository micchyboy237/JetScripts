# See which process owns each tunnel
for i in $(seq 0 5); do
  echo "=== utun$i ==="
  lsof -i -n | grep utun$i
done


# Header for the table
printf "%-10s %-8s %-20s %-10s %-s\n" "INTERFACE" "PID" "PROCESS" "USER" "CONNECTION"
printf "%-10s %-8s %-20s %-10s %-s\n" "----------" "------" "--------------------" "----------" "----------"

# See which process owns each tunnel
for i in $(seq 0 5); do
  output=$(lsof -i -n 2>/dev/null | grep "utun$i")
  
  if [[ -z "$output" ]]; then
    printf "%-10s %-8s %-20s %-10s %-s\n" "utun$i" "-" "No process" "-" "-"
  else
    echo "$output" | while read -r line; do
      # Parse lsof output columns
      process=$(echo "$line" | awk '{print $1}')
      pid=$(echo "$line" | awk '{print $2}')
      user=$(echo "$line" | awk '{print $3}')
      connection=$(echo "$line" | awk '{for(i=9;i<=NF;i++) printf "%s ", $i; print ""}' | xargs)
      
      printf "%-10s %-8s %-20s %-10s %-s\n" "utun$i" "$pid" "$process" "$user" "$connection"
    done
  fi
done
