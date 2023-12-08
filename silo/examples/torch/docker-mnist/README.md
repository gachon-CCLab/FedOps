## Start Silo Client Docker

- **run docker compose**
```bash
# run docker compose
$ docker compose up

# or if you want to monitor client, run below
$ docker compose -f docker-dist/docker-compose.yml up
```
When you want to apply code changes, do `docker compose up --build`.  


- **terminate client**
```bash
# press Ctrl + C to escape
$ docker compose down
```   

*(Tested on Docker Desktop 4.15.0; Docker Compose version 2.13.0; arm64 processor(macOS))*   
*(Tested on x86-64 processor)*



## Start python or shell
- **0. Set client Env**

    ```bash
    # Install fedops libray in your python env(conda)
    pip install -r requirements.txt

    # Install torch & torchvision for using pytorch
    pip install torch torchvision

    ```

- **1. Create two CLI and run python file**

    ```bash
    # One CLI => client
    python3 client_main.py

    # The other CLI => client_manager
    python3 client_manager_main.py

    ```

- **2. Run shell file**

    ```bash
    sh run_shell_client.sh
    ```


## Support
***For any questions or issues, please contact the FedOps support team at gyom1204@gachon.ac.kr***
