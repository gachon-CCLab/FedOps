import itertools
import json, os
import logging
from collections import Counter
from datetime import datetime

import tensorflow as tf
import numpy as np
# from keras.utils.np_utils import to_categorical # keras==2.8.0
from keras.utils import to_categorical # keras>=2.10.0

# set log format
handlers_list = [logging.StreamHandler()]

# if os.environ["MONITORING"] == '1':
#     handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
# else:
#     pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)



"""
Create your data loader for training/testing local_model.
Keep the value of the return variable for normal operation.
----------------------------------------------------------
dataset example
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



# Load the dataset partitions
# def data_load(all_client_num, FL_client_num, dataset, skewed, skewed_spec, balanced):
#     # Load the dataset
#     if dataset == 'cifar10':
#         (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
#     elif dataset == 'mnist':
#         (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#     elif dataset == 'fashion_mnist':
#         (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#
#     (X_train, y_train), (X_test, y_test) = data_partition(X_train, y_train, X_test, y_test, skewed, skewed_spec,
#                                                           balanced, FL_client_num, all_client_num, dataset)
#
#     # preprocessing
#     # train_features = X_train.astype('float32') / 255.0
#     # test_features = X_test.astype('float32') / 255.0
#
#     # return (train_features, train_labels), (test_features, test_labels)
#     return (X_train, y_train), (X_test, y_test)
#
#
# def data_partition(X_train, y_train, X_test, y_test, skewed, skewed_spec, balanced, FL_client_num, all_client_num,
#                    dataset):
#     np.random.seed(FL_client_num)
#
#     # only balanced/Imbalanced
#     if skewed == False:
#         # Balanced dataset
#         # Dataset size range for each FL Client
#         train_range_size = int(len(y_train) / all_client_num)
#         test_range_size = int(len(y_test) / all_client_num)
#
#         train_first_size = train_range_size * FL_client_num
#         test_first_size = test_range_size * FL_client_num
#
#         train_next_size = train_first_size + train_range_size
#         test_next_size = test_first_size + test_range_size
#
#         if balanced == True:
#             (X_train, y_train) = X_train[train_first_size:train_next_size], y_train[train_first_size:train_next_size]
#             (X_test, y_test) = X_test[test_first_size:test_next_size], y_test[test_first_size:test_next_size]
#
#         else:
#             # Imbalanced dataset
#             # Random FL Client dataset size => Imbalanced
#             train_size_st = np.random.randint(train_first_size, train_first_size + 100)
#             test_size_st = np.random.randint(test_first_size, test_first_size + 100)
#             train_size_end = np.random.randint(train_size_st, train_next_size)
#             test_size_end = np.random.randint(test_size_st, test_next_size)
#
#             (X_train, y_train) = X_train[train_size_st:train_size_end], y_train[train_size_st:train_size_end]
#             (X_test, y_test) = X_test[test_size_st:test_size_end], y_test[test_size_st:test_size_end]
#
#         if dataset == 'cifar10':
#             pass
#
#         else:
#             # 28X28 -> 32X32
#             # Pad with 2 zeros on left and right hand sides-
#             X_train = np.pad(X_train[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')
#             X_test = np.pad(X_test[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')
#
#             # Change the shape of the array to set the number of dimensions to 3
#             # When setting the transfer learning model input value, the dimension must be set to 3
#             X_train = tf.expand_dims(X_train, axis=3, name=None)
#             X_test = tf.expand_dims(X_test, axis=3, name=None)
#             X_train = tf.repeat(X_train, 3, axis=3)
#             X_test = tf.repeat(X_test, 3, axis=3)
#
#
#     # balanced skewed/ imbalanced skewed
#     else:
#         print(f'skewed_spec: {skewed_spec}')
#         (X_train, y_train), (X_test, y_test) = skewed_partition(X_train, y_train, X_test, y_test, skewed_spec, balanced,
#                                                                 FL_client_num, all_client_num, dataset)
#
#     return (X_train, y_train), (X_test, y_test)
#
#
# # Provide different label to each client
# def skewed_label(FL_client_num, skewed_spec):
#     print(f'FL_client_num: {FL_client_num}, skewed_spec: {skewed_spec}')
#     if skewed_spec == 'skewed_one':
#         if FL_client_num == 0:
#             labels = 0
#         elif FL_client_num == 1:
#             labels = 1
#         elif FL_client_num == 2:
#             labels = 2
#         elif FL_client_num == 3:
#             labels = 3
#         elif FL_client_num == 4:
#             labels = 4
#     elif skewed_spec == 'skewed_two':
#         if FL_client_num == 0:
#             labels = [0, 1]
#         elif FL_client_num == 1:
#             labels = [2, 3]
#         elif FL_client_num == 2:
#             labels = [4, 5]
#         elif FL_client_num == 3:
#             labels = [6, 7]
#         elif FL_client_num == 4:
#             labels = [8, 9]
#     elif skewed_spec == 'skewed_three':
#         if FL_client_num == 0:
#             labels = [0, 1, 2]
#         elif FL_client_num == 1:
#             labels = [2, 3, 4]
#         elif FL_client_num == 2:
#             labels = [4, 5, 6]
#         elif FL_client_num == 3:
#             labels = [6, 7, 8]
#         elif FL_client_num == 4:
#             labels = [0, 8, 9]
#
#     # print(f'labels: {labels}')
#
#     return labels
#
#
# # Imbalanced/one class Skewed
# def skewed_partition(X_train, y_train, X_test, y_test, skewed_spec, balanced, FL_client_num, all_client_num, dataset):
#     np.random.seed(FL_client_num)
#
#     labels = skewed_label(FL_client_num, skewed_spec)
#
#     # select label index
#     train_indexs = []
#     test_indexs = []
#     if skewed_spec == 'skewed_one':
#         train_indexs.append(np.where(y_train == labels)[0])
#         test_indexs.append(np.where(y_test == labels)[0])
#     else:
#         for label in labels:
#             train_indexs.append(np.where(y_train == label)[0])
#             test_indexs.append(np.where(y_test == label)[0])
#
#     train_index_list = list(itertools.chain(*train_indexs))
#     test_index_list = list(itertools.chain(*test_indexs))
#
#     #     print(f'all train_list: {len(train_index_list)}')
#
#     # Dataset size range for each FL Client
#     train_range_size = int(len(train_index_list) / all_client_num)
#     train_first_size = train_range_size * FL_client_num
#     train_next_size = train_first_size + train_range_size
#     #     print(f'init train_range_size: {train_range_size}')
#     #     print(f'init train_first_size: {train_first_size}')
#     #     print(f'init train_next_size: {train_next_size}')
#
#     test_range_size = int(len(test_index_list) / all_client_num)
#     test_first_size = test_range_size * FL_client_num
#     test_next_size = test_first_size + test_range_size
#     #     print(f'init test_range_size: {test_range_size}')
#     #     print(f'init test_first_size: {test_first_size}')
#     #     print(f'init test_next_size: {test_next_size}')
#
#     if balanced == True:
#         # label index list화 및 정렬 => label 분포있게 추출 가능
#         train_index_list = np.sort(train_index_list)
#         test_index_list = np.sort(test_index_list)
#
#         # balanced label
#         (X_train, y_train) = X_train[train_index_list], y_train[train_index_list]
#         (X_test, y_test) = X_test[test_index_list], y_test[test_index_list]
#
#         (X_train, y_train) = X_train[train_first_size:train_next_size], y_train[train_first_size:train_next_size]
#         (X_test, y_test) = X_test[test_first_size:test_next_size], y_test[test_first_size:test_next_size]
#
#     else:
#         # set Skewed/Imbalanced data random index
#         random.shuffle(train_index_list)
#         random.shuffle(test_index_list)
#
#         # imbalanced label
#         (X_train, y_train) = X_train[train_index_list], y_train[train_index_list]
#         (X_test, y_test) = X_test[test_index_list], y_test[test_index_list]
#
#         # Imbalanced/Skewed dataset
#         # Random FL Client dataset size => Imbalanced
#         train_size_st = np.random.randint(train_first_size, train_first_size + 300)
#         train_size_end = np.random.randint(train_size_st + 300, train_next_size - 300)
#         # print(f'random train_size_st: {train_size_st}')
#         # print(f'random train_size_end: {train_size_end}')
#
#         test_size_st = np.random.randint(test_first_size, test_first_size + 100)
#         test_size_end = np.random.randint(test_size_st, test_next_size - 100)
#         # print(f'random test_size_st: {test_size_st}')
#         # print(f'random test_size_end: {test_size_end}')
#
#         (X_train, y_train) = X_train[train_size_st:train_size_end], y_train[train_size_st:train_size_end]
#         (X_test, y_test) = X_test[test_size_st:test_size_end], y_test[test_size_st:test_size_end]
#
#     #         print(f'random next train: {y_train}')
#
#     if dataset == 'cifar10':
#         pass
#
#     else:
#         # 28X28 -> 32X32
#         # Pad with 2 zeros on left and right hand sides-
#         X_train = np.pad(X_train[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')
#         X_test = np.pad(X_test[:, ], ((0, 0), (2, 2), (2, 2)), 'constant')
#
#         X_train = tf.expand_dims(X_train, axis=3, name=None)
#         X_test = tf.expand_dims(X_test, axis=3, name=None)
#         X_train = tf.repeat(X_train, 3, axis=3)
#         X_test = tf.repeat(X_test, 3, axis=3)
#
#     return (X_train, y_train), (X_test, y_test)