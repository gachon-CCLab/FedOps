#!/bin/sh

# start the container stack
# (assumes the caller has permission to do this)
docker-compose -f docker-compose.yml up -d --build
