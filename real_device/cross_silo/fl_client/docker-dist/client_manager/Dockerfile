FROM python:3.8-alpine
WORKDIR /app
ENV APP=client_manager.py
ENV RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY ./docker-dist/client_manager/requirements.txt requirements.txt
RUN pip install -r requirements.txt
# RUN git clone https://github.com/gachon-CCLab/fl-client.git
# WORKDIR /app/fl-client
EXPOSE 8003
COPY . .
ENTRYPOINT [ "python3", "/app/client_manager/client_manager.py", "docker-dist" ]