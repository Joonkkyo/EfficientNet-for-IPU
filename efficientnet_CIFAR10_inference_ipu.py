import os
import sys
from skimage.io import imread
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10 # cifar10 데이터셋을 가져옴
import tensorflow.keras as tfkeras # tensorflow 2.0 version부터 편입된 keras 모듈 사용
from skimage.transform import resize
from tensorflow.python.keras.layers import Input # 모델의 input layer를 정의하는 메소드인 Input을 가져옴
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.python import dtypes # data type을 가져오기 위한 메소드인 dtypes 사용
from model import inject_ipukeras_modules
from model.efficientnet_model_ipu import EfficientNetB0 # model 폴더 내의 efficientnet_model_ipu.py내부에 있는 EfficientNetB0 함수를 가져온다.
from tensorflow.python import ipu # ipu 환경에서 모델을 동작시키기 위한 ipu 모듈 사용

MAP_INTERPOLATION_TO_ORDER = {
    "nearest": 0,
    "bilinear": 1,
    "biquadratic": 2,
    "bicubic": 3,
}


def center_crop_and_resize(image, image_size, crop_padding=32, interpolation="bicubic"):
    assert image.ndim in {2, 3}
    assert interpolation in MAP_INTERPOLATION_TO_ORDER.keys()

    h, w = image.shape[:2]

    padded_center_crop_size = int(
        (image_size / (image_size + crop_padding)) * min(h, w)
    )
    offset_height = ((h - padded_center_crop_size) + 1) // 2
    offset_width = ((w - padded_center_crop_size) + 1) // 2

    image_crop = image[
                 offset_height: padded_center_crop_size + offset_height,
                 offset_width: padded_center_crop_size + offset_width,
                 ]
    resized_image = resize(
        image_crop,
        (image_size, image_size),
        order=MAP_INTERPOLATION_TO_ORDER[interpolation],
        preserve_range=True,
    )

    return resized_image

## ipu.keras.Model을 안에서 사용할 수 있도록 설정 (efficientnet 내부에는 생성자 함수(__init__.py)에서 여러 환경에 맞춰 실행할 수 있도록 내부 메타 데이터를 튜닝할 수 있다.)
EfficientNetB0 = inject_ipukeras_modules(EfficientNetB0) # tensorflow.keras 환경에서 구현된 모델을 ipu.keras 환경에서도 동작할 수 있도록 설정
# Define the model
def get_model():
    input_tensor = Input(shape=(32, 32, 3), dtype=dtypes.float32) # Cifar10 데이터셋의 shape에 맞게 input tensor 설정
    return EfficientNetB0(
        include_top=True, # fully connected(fc)층을 포함시킬지 아닐지를 결정하는 옵션, True는 포함, False는 포함시키지 않음 
        weights=None,
        #input_tensor=input_tensor,
        # models=ipu.keras.Model, # 해당 모델이 ipu.keras에서 돌아갈 수 있도록 설정한다.
        pooling='max',
        classes=10
    )

if __name__ == '__main__':
    cfg = ipu.utils.create_ipu_config() # ipu에 대한 configuration 설정 시작
    cfg = ipu.utils.auto_select_ipus(cfg, 2) # 실행할 ipu 설정
    ipu.utils.configure_ipu_system(cfg) # ipu 세팅 완료
    strategy = ipu.ipu_strategy.IPUStrategy() 

    with strategy.scope():
        model = get_model() # 세팅해놓은 모델을 가져온다.
        checkpoint_path = "checkpoint/effn.ckpt"
        model.load_weights(checkpoint_path)
        # (_, _), (x_test, y_test) = cifar10.load_data()
        # x_test = np.expand_dims(x_test, 0)
        # print(x_test.shape)
        # test_dataset = tf.data.Dataset.from_tensor_slices(x_test) \
        # .map(lambda x : (tf.cast(x, tf.float32)))
     
        model.summary()
        image = imread('./image/panda.jpg')
        image_size = model.input_shape[1]
        print(model.input_shape)
        x = center_crop_and_resize(image, image_size=image_size)
        # # x = preprocess_input(x)
        x = np.expand_dims(x, 0)
        print(x.shape)
        checkpoint_path = "checkpoint/effn.ckpt"
        print(x)
        y = model.predict(x)
        decode_predictions(y)
