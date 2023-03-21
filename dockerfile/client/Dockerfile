FROM nvidia/cuda:11.7.0-base-ubuntu22.04

RUN apt-get -y update

RUN apt-get -y install gcc git
RUN apt-get -y install python3.8-dev
RUN apt-get -y install python3-pip

ENV TZ=Asia/Seoul

# Define the arguments
ENV REPO_URL=https://github.com/gachon-CCLab/FedOps.git
ENV GIT_TAG=main
ENV GIT_DIR=FedOps/k8s_simulation/client

COPY python.sh python.sh

RUN chmod +x /python.sh

# Set the working directory to the Git directory
WORKDIR /app

ENTRYPOINT ["/bin/sh", "-c", "/python.sh"]
