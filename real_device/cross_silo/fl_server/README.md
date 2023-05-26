# FL Server

## FL Server Code Structure
**config.yaml**:
- fl_server: Set the FL server hyperparameters
- aggregation: Set the FL aggregation(FedAvg) algorithm hyperparameters
- data: Set the data name for using to FL task

**init_gl_model.py**:
- Implement the ***init_gl_model.py*** module to initialize the global model.
- This module should include the necessary code for initializing the global model.

**server_task.py**:
- To register the data used for evaluating the global model and initialize the global model, execute the ***server_task.py*** script.
