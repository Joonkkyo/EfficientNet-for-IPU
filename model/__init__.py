# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
from .__version__ import __version__

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def inject_keras_modules(func):
    import keras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        return func(*args, **kwargs)

    return wrapper


def inject_tfkeras_modules(func):
    import tensorflow.keras as tfkeras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = tfkeras.backend
        kwargs['layers'] = tfkeras.layers
        kwargs['models'] = tfkeras.models
        kwargs['utils'] = tfkeras.utils
        return func(*args, **kwargs)

    return wrapper


## ipu.keras.Model을 안에서 사용할 수 있도록 설정 (efficientnet 내부에는 생성자 함수(__init__.py)에서 여러 환경에 맞춰 실행할 수 있도록 내부 메타 데이터를 튜닝할 수 있다.)
def inject_ipukeras_modules(func):
    import tensorflow.keras as tfkeras
    from tensorflow.python import ipu
    @functools.wraps(func) # wraps() are tools to help write wrapper functions that can handle naive introspection
    def wrapper(*args, **kwargs): # wrapper를 통해 keras의 meta data에 접근하여 ipu.keras.Model이 기존 keras와 호환이 되도록 설정한다.
        kwargs['backend'] = tfkeras.backend # backend 환경은 기존 keras에서 작동
        kwargs['layers'] = tfkeras.layers # layer에 대한 모듈을 사용 
        kwargs['models'] = ipu.keras # keras의 내부에서 사용할 모델을 ipu 환경으로 실행
        kwargs['utils'] = tfkeras.utils 
        return func(*args, **kwargs)
    return wrapper


def init_keras_custom_objects():
    import keras
    from . import model

    custom_objects = {
        'swish': inject_keras_modules(model.get_swish)(),
        'FixedDropout': inject_keras_modules(model.get_dropout)()
    }
    
    try:
        keras.utils.generic_utils.get_custom_objects().update(custom_objects)
    except AttributeError:
        keras.utils.get_custom_objects().update(custom_objects)


def init_tfkeras_custom_objects():
    import tensorflow.keras as tfkeras
    from . import model

    custom_objects = {
        'swish': inject_tfkeras_modules(model.get_swish)(),
        'FixedDropout': inject_tfkeras_modules(model.get_dropout)()
    }

    tfkeras.utils.get_custom_objects().update(custom_objects)
