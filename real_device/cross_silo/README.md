# FedOps on Real Device

This guide provides step-by-step instructions on executing FedOps, a Federated Learning Lifecycle Management Operations framework.

## Steps

1. ***Start by cloning the FedOps.***

```shell
git clone https://github.com/gachon-CCLab/FedOps.git && mv FedOps/real_device/cross_silo . && rm -rf FedOps && cd cross_silo
```

This will create a new directory called `cross_silo` containing the following files:
 
```shell
-- fl_client
-- fl_server
-- README.md
```

2. ***Create your Git repository and add FL server code.***
   - Set up your own Git repository and add the FL server code from the fl_server directory. 
   This code will be used to deploy and run the FL server in CCL microk8s environment.
   <br></br>

3. ***Create FL task on FedOps web interface.***
   - Use the FedOps web interface to create your FL task. 
   Specify the Git repository address for your FL server code.
   <br></br>

4. ***Customize the FedOps example code.***
   - Customize the FedOps example code to align with your FL task.
   * [FedOps Client](https://github.com/gachon-CCLab/FedOps/tree/main/real_device/cross_silo/fl_client)
   - Client:
     - Configure settings in config.yaml for task ID, data, and WandB information.
     - Implement client_data.py for data handling.
     - Build client_model.py for local model specifications.
     - Register data and model, and run the client using client_task.py.

   - Server:
     * [FedOps Server](https://github.com/gachon-CCLab/FedOps/tree/main/real_device/cross_silo/fl_server)
     - Configure settings in config.yaml for FL/Aggregation hyperparameters and data information.
     - Implement init_gl_model.py to initialize the global model.
     - Register data (for evaluating the global model) and initialize the global model in server_task.py.
     <br></br>

5. ***Run the clients.***
   - Choose either Docker or shell(localhost) to run the clients. 
   Detailed instructions on running the clients can be found in the FedOps documentation.
   <br></br>

6. ***Initiate the FL task.***
   - Select the desired clients on the FedOps web interface and initiate the FL task by clicking the "FL start" button.
   <br></br>

7. ***Monitor and manage the FL task***
   - Monitor the performance of local and global models, 
   and manage/download the global model as the FL task administrator through the FedOps web interface.
   <br></br>

8. ***Monitor data and local model performance*** 
   - Monitor the health of data and local model performance as the device administrator through the designated WandB.
   <br></br>

## Support
***For any questions or issues, please contact the FedOps support team at tpah20@gachon.ac.kr***