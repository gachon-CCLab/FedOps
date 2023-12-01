FROM python:3.10-alpine
WORKDIR /app
ENV APP=client_manager_main.py
ENV RUN_HOST=0.0.0.0

RUN apk add --no-cache gcc musl-dev linux-headers

COPY . .

RUN pip install -r ./docker-dist/client_manager/requirements.txt

EXPOSE 8004
ENTRYPOINT [ "python3", "client_manager_main.py", "docker=true"]