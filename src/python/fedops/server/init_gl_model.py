import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import inspect

"""
Create model method to build init global model.
And, you have to return values that is model & model_name
Model name is method function name.
Refer to this code structure.
"""

def CNN():

    model_name = inspect.currentframe().f_code.co_name # function name

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]

    # model 생성
    model = Sequential()

    # Convolutional Block (Conv-Conv-Pool-Dropout)
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classifying
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=METRICS)

    return model , model_name


def VGG16():

    model_name = inspect.currentframe().f_code.co_name  # function name

    # keras.VGG16
    model_VGG16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    for layer in model_VGG16.layers:
        layer.trainable = False

    # Organize the layers in the same way as the structure of VGG16 above.
    x = tf.keras.layers.Flatten()(model_VGG16.output)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1000, activation='relu')(x)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

    # optimizer -> 'adam'
    # loss -> sparse_categorical_crossentropy
    model = tf.keras.Model(inputs=model_VGG16.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    # model.summary()

    return model, model_name


def ResNet50():

    model_name = inspect.currentframe().f_code.co_name  # function name

    # keras.resnet50
    model_Res = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    for layer in model_Res.layers:
        layer.trainable = False

    # Determining output class after flattening in transfer learning model
    x = tf.keras.layers.Flatten()(model_Res.output)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=model_Res.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    # model.summary()

    return model, model_name