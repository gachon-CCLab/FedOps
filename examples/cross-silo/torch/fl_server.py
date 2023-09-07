from fedops.server import app
from fedops.server import server_utils
import fl_model
import fl_data


if __name__ == "__main__":
    # Read server config file
    config_file_path = '../config.yaml'
    config = server_utils.read_config(config_file_path)

    # Dataset Name
    # dataset = config['data']['name']
    
    """
    Build initial global model based on dataset name.
    Set the initial global model you created in fl_model.py to match the dataset name.
    """
    # Build init global model using torch
    model = fl_model.DNN(input_dim=6, num_classes=5)
    criterion, optimizer = fl_model.set_model_parameter(model)
    model_name = model.model_name
    gl_test_torch = fl_model.test_torch() # set torch test
    
    # model, model_name = fl_model.CNN() # Build init global model using tensorflow
    
    # Check tensorflow or torch model
    model_type = server_utils.identify_model(model)

    # Load validation data for evaluating global model
    gl_val_loader = fl_data.gl_model_torch_validation(batch_size=config['server']['fl_round']['batch_size']) # torch
    # x_val, y_val = fl_data.gl_model_tensorflow_validation() # tensorflow
    

    # Start fl server
    fl_server = app.FLServer(config=config, model=model, model_name=model_name, model_type=model_type,criterion=criterion, 
                             optimizer=optimizer, gl_val_loader=gl_val_loader, test_torch=gl_test_torch) # torch
    # fl_server = app.FLServer(config=config, model=model, model_name=model_name, x_val=x_val, y_val=y_val) # tensorflow
    fl_server.start()

