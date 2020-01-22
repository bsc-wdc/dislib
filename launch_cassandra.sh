docker network create --driver bridge cassandra_bridge
# launch Cassandra
CASSANDRA_ID=$(docker run --rm --network=cassandra_bridge -d cassandra)
sleep 30
CASSANDRA_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${CASSANDRA_ID}")
# add environment variable CONTACT_NAMES needed by Hecuba
export CONTACT_NAMES=$CASSANDRA_IP
echo "Using Cassandra host: $CONTACT_NAMES"
