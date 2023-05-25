# FROM nvidia/cuda:11.7.0-base-ubuntu22.04
FROM nvidia/cuda:11.7.0-base-ubuntu22.04
WORKDIR /app
ENV APP=client.py
ENV RUN_HOST=0.0.0.0
# RUN apt-get -y update \
#     && apt-get install -y software-properties-common \
#     && apt-get -y update \
#     && add-apt-repository universe
RUN apt-get -y update
# RUN apt-get -y install git
RUN apt-get -y install gcc
RUN apt-get -y install python3.8-dev
RUN apt-get -y install python3-pip
COPY ./docker-dist/client/requirements.txt requirements.txt
RUN pip install -r requirements.txt
# RUN git clone https://github.com/gachon-CCLab/fl-client.git
# WORKDIR /app/fl-client
EXPOSE 8002
COPY . .
ENTRYPOINT [ "python3", "/app/client/client.py", "docker-dist" ]