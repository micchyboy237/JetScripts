docker stop $(docker ps -q)
docker system prune -a -f
docker volume prune -f
docker image prune -a -f
rm -rf ~/Library/Caches/com.docker.docker/*
rm -rf ~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw
open -a Docker && osascript -e 'quit app "Docker"' && open -a Docker
