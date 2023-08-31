import json
import logging
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset, Dataset

import pandas as pd
from sklearn.model_selection import train_test_split

# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


"""
Create your data loader for training/testing local_model.
Keep the value of the return variable for normal operation.
----------------------------------------------------------
dataset example
"""


# Pytorch version

# Define a custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, targets, indices=None):
        self.data = data[indices]
        self.targets = targets[indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    

def load_partition(dataset, FL_client_num, validation_split, label_count, batch_size):

    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')

    file_path = f"/home/ccl/Desktop/FedOps/examples/fl_data/triage_group_homo_{FL_client_num}.csv"

    df = pd.read_csv(file_path)

    # Select the features and target variable
    X = df[['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']]
    y = df['acuity']  # Replace 'target_variable' with the appropriate column name

    # Step 1: Convert 'y' to one-hot encoding
    # Because of Multi label
    y_one_hot = pd.get_dummies(y)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_one_hot.values, dtype=torch.float32)
    # y_tensor = torch.tensor(y.values, dtype=torch.long)  # Use integer labels


    # Train-Test split
    train_indices, val_indices = train_test_split(range(len(df)), test_size=validation_split, random_state=42)
    
    # Create train and test datasets using the custom Dataset class
    train_dataset = CustomDataset(X_tensor, y_tensor, train_indices)
    test_dataset = CustomDataset(X_tensor, y_tensor, val_indices)

    # TrainSet/ValidationSet
    n_valset = int(len(train_dataset) * validation_split)

    valset = Subset(train_dataset, range(0, n_valset))
    trainset = Subset(
        train_dataset, range(n_valset, len(train_dataset))
    )

    # DataLoader for training and validation, test
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # data check => IID VS Non IID
    y_label_counter = Counter(y)

    # check client data(label) => non-iid
    for i in range(1,label_count+1):
        data_check_dict = {"client_num": int(FL_client_num), "label_num": i, "data_size": int(y_label_counter[i])}
        data_check_json = json.dumps(data_check_dict)
        logging.info(f'data_check - {data_check_json}')

    return train_loader, val_loader, test_loader, y_label_counter

# Tensorflow version
"""
# load dataset
def load_partition(dataset, FL_client_num, label_count):
    # Load the dataset
    if dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    if dataset == 'cifar10':
        pass

    else:  # Since the models of MNIST and FashionMNIST are transfer learning models, they are set in three dimensions.
        # 28X28 -> 32X32
        # Pad with 2 zeros on left and right hand sides-
        X_train = np.pad(X_train[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')
        X_test = np.pad(X_test[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')

        X_train = tf.expand_dims(X_train, axis=3, name=None)
        X_test = tf.expand_dims(X_test, axis=3, name=None)
        X_train = tf.repeat(X_train, 3, axis=3)
        X_test = tf.repeat(X_test, 3, axis=3)

    # Split dataset by client_num
    (X_train, y_train) = X_train[FL_client_num * 2000:(FL_client_num + 1) * 2000], y_train[FL_client_num * 2000:(FL_client_num + 1) * 2000]
    (X_test, y_test) = X_test[FL_client_num * 1000:(FL_client_num + 1) * 1000], y_test[FL_client_num * 1000:(FL_client_num + 1) * 1000]

    # one-hot encoding class
    # Since each client has different labels, the total number of labels must be matched
    train_labels = to_categorical(y_train, label_count)
    test_labels = to_categorical(y_test, label_count)

    # 전처리
    train_features = X_train.astype('float32') / 255.0
    test_features = X_test.astype('float32') / 255.0


    # data check => IID VS Non IID
    # array -> list
    y_list = y_train.tolist()
    y_train_label = list(itertools.chain(*y_list))
    y_label_counter = Counter(y_train_label)

    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')

    # check client data(label) => non-iid
    for i in range(label_count):
        data_check_dict = {"client_num": int(FL_client_num), "label_num": i, "data_size": int(y_label_counter[i])}
        data_check_json = json.dumps(data_check_dict)
        logging.info(f'data_check - {data_check_json}')

    return (train_features, train_labels), (test_features, test_labels), y_label_counter
"""



"""
Create your data loader that matches the dataset name for evaluating global model.
Keep the value of the return variable for normal operation.
----------------------------------------------------------
dataset example
"""

# Pytorch version
def gl_model_torch_validation(batch_size):
    file_path = f"/home/ccl/Desktop/FedOps/examples/fl_data/triage_group_homo_6.csv"

    df = pd.read_csv(file_path)

    # Select the features and target variable
    X = df[['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']]
    y = df['acuity']  # Replace 'target_variable' with the appropriate column name

    # Step 1: Convert 'y' to one-hot encoding
    # Because of Multi label
    y_one_hot = pd.get_dummies(y)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_one_hot.values, dtype=torch.float32)
    
    val_dataset = CustomDataset(X_tensor, y_tensor, X.index)
    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)

    
    return gl_val_loader


# Tensorflow version
"""
def gl_model_tensorflow_validation()
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
"""

