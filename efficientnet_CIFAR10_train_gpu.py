import tensorflow as tf
import time
import numpy as np
import efficientnet.keras
from keras import layers
from keras import models


def get_model():
    return efficientnet.keras.EfficientNetB0(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    pooling='max',
    classes=10
    )


def get_dataset():
    cifar = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar.load_data()

    mean = [0,0,0]
    std = [0,0,0]
    newX_train = np.ones(x_train.shape)
    newX_test = np.ones(x_test.shape)
    for i in range(3):
        mean[i] = np.mean(x_train[:,:,:,i])
        std[i] = np.std(x_train[:,:,:,i])
        
    for i in range(3):
        newX_train[:,:,:,i] = x_train[:,:,:,i] - mean[i]
        newX_train[:,:,:,i] = newX_train[:,:,:,i] / std[i]
        newX_test[:,:,:,i] = x_test[:,:,:,i] - mean[i]
        newX_test[:,:,:,i] = newX_test[:,:,:,i] / std[i]        
        
    x_train = newX_train
    x_test = newX_test

    print ("mean after normalization:", np.mean(x_train))
    print ("std after normalization:", np.std(x_train))
    print(x_train.max())
    print(x_train.shape, x_train.dtype)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    ## model initialization 
    gpus = tf.config.experimental.list_logical_devices("GPU")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print('\n\nRunning on multiple GPUs ', [gpu.name for gpu in gpus])
    else:
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('\n\nRunning on single GPU ', gpus[0].name)
        
    model = get_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    x_train, y_train, x_test, y_test = get_dataset()

    start = time.time()
    model.fit(x_train, y_train, epochs=30, batch_size=12)
    end = time.time()

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Loss : {0}, Accuracy : {1}'.format(test_loss, test_acc))
    print('Elapsed time :', end - start)
     # batch size = 12, epoch = 30, elapsed time : 5547s, accuracy : 0.7216
