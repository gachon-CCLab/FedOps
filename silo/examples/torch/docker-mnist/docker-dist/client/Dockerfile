# FROM nvidia/cuda:11.7.0-base-ubuntu22.04
FROM python:3.10
WORKDIR /app
ENV APP=client_main.py
ENV RUN_HOST=0.0.0.0
ENV MODE=docker


COPY . .

RUN pip install -r ./docker-dist/client/requirements.txt

EXPOSE 8003

ENTRYPOINT [ "python3", "client_main.py", "docker=true"]
