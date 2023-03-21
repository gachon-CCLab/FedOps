#!/bin/sh

# start the container stack
# (assumes the caller has permission to do this)
docker-compose -f docker-compose-monitoring.yml up -d --build

# wait for the service to be ready
while ! curl --fail --silent --head http://localhost:5601; do
  sleep 1
done

# open the browser (Mac)
open http://localhost:5601

# open the browser (Window)
# start http://localhost:5601

# Source: (https://stackoverflow.com/a/70463577/20211370)