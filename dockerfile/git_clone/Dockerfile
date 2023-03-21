FROM python:3.8-alpine

# Install Git
RUN apk update && apk add --no-cache git

ENV TZ=Asia/Seoul

# Define the arguments
ENV REPO_URL=https://github.com/gachon-CCLab/FedOps.git
ENV GIT_TAG=main
ENV GIT_DIR=FedOps/server_manager

COPY python.sh python.sh

RUN chmod +x /python.sh

# Set the working directory to the Git directory
WORKDIR /app

ENTRYPOINT ["/bin/sh", "-c", "/python.sh"]
