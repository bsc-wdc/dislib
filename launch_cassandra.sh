docker network create --driver bridge cassandra_bridge
# launch Cassandra
CASSANDRA_ID=$(docker run --rm --network=cassandra_bridge -d cassandra)
sleep 30
CASSANDRA_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${CASSANDRA_ID}")
# connect dislib container to Cassandra container
docker network connect cassandra_bridge dislib
# add environment variable CONTACT_NAMES needed by Hecuba
docker exec -d dislib /bin/bash -c 'export CONTACT_NAMES=${$1}' "$CASSANDRA_IP"

