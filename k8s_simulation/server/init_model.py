import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense


def model_cnn():
    # 모델 및 메트릭 정의
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
        
        return model


def model_VGG16():
    # keras.VGG16 전이학습모델 가져오기
    model_VGG16 = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False,input_shape = (32,32,3))
    for layer in model_VGG16.layers:
        layer.trainable = False

    # 위의 VGG16의 구조와 동일하게 레이어를 구성
    x = tf.keras.layers.Flatten()(model_VGG16.output)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1000, activation='relu')(x)
    predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)

    # 옵티마이저와 손실함수를 설정하고 정확도 매트릭스가 나오게 컴파일
    # optimizer -> 'adam'
    # loss -> sparse_categorical_crossentropy
    model = tf.keras.Model(inputs = model_VGG16.input, outputs = predictions)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    # model.summary()

    return model


def model_ResNet50():

    # keras.resnet50 전이학습모델 가져오기
    model_Res = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False,input_shape = (32,32,3))
    for layer in model_Res.layers:
        layer.trainable = False
    
    # 전이학습모델에서 Flatten 후 출력 CLASS 정하기
    x = tf.keras.layers.Flatten()(model_Res.output)
    predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)

    # 모델 학습 후 summary로 확인
    model = tf.keras.Model(inputs = model_Res.input, outputs = predictions)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    # model.summary()

    return model