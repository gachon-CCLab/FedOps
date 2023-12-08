# FedOps Simulation

This guide provides step-by-step instructions on executing FedOps Simulation

## Baseline
 
```shell
- Baseline
   - main.py
   - models.py
   - data_preparation.py
   - utils.py
   - requirements.txt
   - conf
      -  config.yaml
```
Please note that some baselines might include additional files (e.g. a `requirements.txt`) or a hierarchy of `.yaml` files for [Hydra](https://hydra.cc/).


## Steps

1. ***Start by cloning the FedOps.***

- Pytorch (CIFAR100 example)
   ```shell
   git clone https://github.com/gachon-CCLab/FedOps.git && mv FedOps/simulation/examples/CIFAR100 . && rm -rf FedOps
   ```
- Set client env

   ```bash
   # Install libraries in your python env(conda)
   pip install -r requirements.txt
   ```

2. ***Customize the FedOps Simulation example code.***
   - Customize the FedOps simulation example code to align with your FL task.

   - Data & Model:
      - Prepare your data in `data_preparation.py` and define your model in `models.py`.
      - Enter your data and model info in `conf/config.yaml`.

   - Simulation:
     - Register your custom data and model in `main.py`.
     - In addition to data and model, you can change configuration information for FL (refer to the conf/config.yaml file for details).
     - And run this python file
         ```bash
         python3 main.py
         ```
     
     
3. ***Monitor global model performance*** 
   - Monitor a global model performance through the designated WandB.(You need to set up wandb to do this)
   <br></br>

## Support
***For any questions or issues, please contact the FedOps support team at gyom1204@gachon.ac.kr***
