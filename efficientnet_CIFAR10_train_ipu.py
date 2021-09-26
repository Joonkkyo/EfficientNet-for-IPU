import tensorflow as tf
import os
import time
import numpy as np
import functools # 고차 함수(하나 이상의 함수를 입력으로 사용하고 새로운 함수를 반환하는 함수)를 제공해주는 내장 모듈
# import tensorflow.keras as tfkeras # tensorflow 2.0 version부터 편입된 keras 모듈 사용
from model import inject_ipukeras_modules
from model.efficientnet_model_ipu import EfficientNetB0 # model 폴더 내의 efficientnet_model_ipu.py내부에 있는 EfficientNetB0 함수를 가져온다.
from tensorflow.python import dtypes # data type을 가져오기 위한 메소드인 dtypes 사용
from tensorflow.python import ipu # ipu 환경에서 모델을 동작시키기 위한 ipu 모듈 사용
from tensorflow.python.keras.datasets import cifar10 # cifar10 데이터셋을 가져옴
from tensorflow.python.keras.layers import Input # 모델의 input layer를 정의하는 메소드인 Input을 가져옴
from tensorflow.python.keras.optimizer_v2.adam import Adam # 학습시 사용할 optimizer로 Adam을 사용


EfficientNetB0 = inject_ipukeras_modules(EfficientNetB0) # tensorflow.keras 환경에서 구현된 모델을 ipu.keras 환경에서도 동작할 수 있도록 설정
# Define the model
def get_model():
    input_tensor = Input(shape=(32, 32, 3), dtype=dtypes.float32, batch_size=16) # Cifar100 데이터셋의 shape에 맞게 input tensor 설정
    return EfficientNetB0(
        include_top=True, # fully connected(fc)층을 포함시킬지 아닐지를 결정하는 옵션, True는 포함, False는 포함시키지 않음 
        weights=None,
        input_tensor=input_tensor,
        pooling='max',
        classes=10
    )


# Define the dataset
def get_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Data normalization
    mean=[0,0,0]
    std=[0,0,0]
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

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .repeat() \
        .map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32))) \
        .batch(16, drop_remainder=True)
        
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .repeat() \
        .map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32))) \
        .batch(16, drop_remainder=True)

    return train_dataset, test_dataset


if __name__ == '__main__':
    # Configure IPUs
    cfg = ipu.utils.create_ipu_config() # ipu에 대한 configuration 설정 시작
    cfg = ipu.utils.auto_select_ipus(cfg, 2) # 실행할 ipu 설정
    ipu.utils.configure_ipu_system(cfg) # ipu 세팅 완료

    strategy = ipu.ipu_strategy.IPUStrategy() 
    
    with strategy.scope():
        # 세팅해놓은 데이터셋을 가져온다.
        model = get_model()
        # 세팅한 모델을 compile하는 단계이다. label에 대해 one-hot encoding을 실행하지 않았기 때문에 손실함수는 sparse categorical crossentropy로 설정한다.
        model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # 세팅해놓은 데이터셋을 가져온다.
        train_dataset, test_dataset = get_dataset() 
        
        # checkpoint 파일을 저장할 경로 설정
        checkpoint_path = "checkpoint/effn.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        start = time.time()
        # steps_per_epoch = the number of data / batch size (50000 / 12 = 4167)
        # callback을 이용해 학습하는 동안 weight를 별도로 저장한다.
        model.fit(train_dataset, epochs=30, steps_per_epoch=4167, 
                  callbacks=[tf.keras.callbacks.ModelCheckpoint(
                      filepath=checkpoint_path,
                      save_weights_only=True,
                      verbose=1)])
        end = time.time()
        
        test_loss, test_acc = model.evaluate(test_dataset, steps=100) # 학습된 모델을 test 데이터를 통해 평가한다.
        print('Loss : {0}, Accuracy : {1}'.format(test_loss, test_acc))
        print('Elapsed time :', end - start)
        # batch size = 12, epoch = 30, elapsed time : 3242s, accuracy : 0.7498
