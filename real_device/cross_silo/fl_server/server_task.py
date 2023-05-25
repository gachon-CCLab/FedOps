import tensorflow as tf
import numpy as np
# from keras.utils.np_utils import to_categorical # keras==2.8.0
from keras.utils import to_categorical # keras>=2.10.0


from fedops.server import app
from fedops.server import server_utils
import init_gl_model


"""
Build initial global model based on dataset name.
Set the initial global model you created in init_gl_model.py to match the dataset name.
"""
def build_gl_model(dataset):
    # Build init global model

    if dataset == 'cifar10':
        model, model_name = init_gl_model.CNN()

    elif dataset == 'mnist':
        model, model_name = init_gl_model.ResNet50()

    elif dataset == 'fashion_mnist':
        model, model_name = init_gl_model.VGG16()

    return model, model_name


"""
Create your data loader that matches the dataset name for evaluating global model.
Keep the value of the return variable for normal operation.
----------------------------------------------------------
dataset example
"""
def load_data(dataset):

    if dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    if dataset == 'cifar10':
        pass

    else:
        # 28X28 -> 32X32
        # Pad with 2 zeros on left and right hand sides-
        # X_train = np.pad(X_train[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')
        X_test = np.pad(X_test[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')


        # X_train = tf.expand_dims(X_train, axis=3, name=None)
        X_test = tf.expand_dims(X_test, axis=3, name=None)
        # X_train = tf.repeat(X_train, 3, axis=3)
        X_test = tf.repeat(X_test, 3, axis=3)

    num_classes = 10

    # Dataset for evaluating global model
    x_val, y_val = X_test[9000:10000], y_test[9000:10000]

    # Preprocessing
    # x_val = x_val.astype('float32') / 255.0

    # y(label) one-hot encoding
    y_val = to_categorical(y_val, num_classes)

    return x_val, y_val


if __name__ == "__main__":
    # Read server config file
    config_file_path = './config.yaml'
    config = server_utils.read_config(config_file_path)

    # Dataset Name
    dataset = config['data']['name']

    # Build model
    model, model_name = build_gl_model(dataset)

    # Load Data
    x_val, y_val = load_data(dataset)

    # Start fl server
    fl_server = app.FLServer(config, model, model_name, x_val, y_val)
    fl_server.start()

