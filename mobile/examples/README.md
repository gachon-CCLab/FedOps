# FedOps Mobile

This guide provides step-by-step instructions on executing FedOps Simulation

## Baseline
 
```shell
- mobile_client
   - Flutter
      - android
      - ios

- server
   - __init__.py
   - server_main.py
   - requirements.txt
   - conf
      -  config.yaml
```

## Mobile Client Steps
### Based on Flutter
- ***Preparing***



## FL Server Steps

- ***Server Setting***
     - Configure settings in `conf/config.yaml` for device type(android or ios) and FL Server Aggregation hyperparameters.
     <br></br>

- ***Create your Git repository and add FL server code.***
   - Set up your own Git repository and add the FL server code (`server_main.py, requirementst.txt, conf/config.yaml`) from the server directory. 
   - This code will be used to deploy and run the FL server in CCL k8s environment.
   <br></br>

- ***Create FL task on FedOps web interface.***
   - Use the FedOps web interface to create your FL task. 
   - Specify the Git repository address for your FL server code.
   - Refer [FedOps Mobile Guide](https://gachon-cclab.github.io/docs/user-guide/mobile-guide/)
   <br></br>


## Support
***For any questions or issues, please contact the FedOps support team at gyom1204@gachon.ac.kr (server part) and farxody98@gachon.ac.kr (mobile part)***
