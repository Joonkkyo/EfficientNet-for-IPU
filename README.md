# EfficientNet for the IPU

## Overview
* GPU용 예제 코드를 IPU용 예제 코드로 변환
* 모델 및 학습 관련 코드를 IPU 환경에서 학습할 수 있도록 변환


## 폴더 구조
```
├── model/                                  - IPU 전용 모델 정의
│    ├── __init__.py                        - 모듈 초기화
│    ├── __version__.py                     - 패키지 버전 정의 
│    ├── efficientnet_model_ipu.py          - IPU용 efficientnet 모델
│    ├── efficientnet_model_ipu_param.py    - IPU용 efficientnet 모델 (최적화)
│    └── weights.py                         - 사용자 정의 
│
├── README.md                               - 리드미 파일
├── efficientnet_CIFAR10_inference_ipu.py   - CIFAR10 데이터 추론 
├── efficientnet_CIFAR10_train_gpu.py       - CIFAR10 데이터셋 학습 (GPU)
└── efficientnet_CIFAR10_train_ipu.py       - CIFAR10 데이터셋 학습 (IPU)
```
## 환경 설정
* python 3.6 버전 가상환경 설치 및 활성화
```
virtualenv venvtf21 -p python3.6
source ~/venvtf21/bin/activate
```
* Poplar SDK, gc-tensorflow 설치 및 활성화
```
source ~/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/enable.sh
## AMD CPU 서버인 경우
pip install ~/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/tensorflow-2.1.2+gc2.0.0+35721+f8f638bad2d+amd_znver1-cp36-cp36m-linux_x86_64.whl
## Intel CPU 서버인 경우
pip install ~/poplar_sdk-ubuntu_18_04-2.0.0+481-79b41f85d1/tensorflow-2.1.2+gc2.0.0+35723+f8f638bad2d+intel_skylake512-cp36-cp36m-linux_x86_64.whl
```
## 코드 비교 
### - efficientnet_CIFAR10_train_gpu.py vs efficientnet_CIFAR10_train_ipu.py 

![스크린샷 2021-04-22 오후 3 42 25](https://user-images.githubusercontent.com/12121282/115667905-578e8a00-a381-11eb-9ea0-8bdd2f77a29d.png)
* 왼쪽이 GPU 버전, 오른쪽이 IPU 버전입니다.
* 프로그램 실행에 필요한 파이썬 모듈을 import합니다.
* GPU 버전에서는 keras 모듈에 내장되어있는 EfficientNet 모델을 사용하지만, IPU에서는 파이프라이닝 및 변환 작업을 거친 모델을 사용해야 하기 때문에 별도로 수정한 모델을 import합니다.
* IPU API는 top-level의 tensorflow 모듈에서는 지원되지 않습니다. 따라서 tensorflow에서 ipu를 사용하기 위해서는 tensorflow.python 모듈에 있는 함수를 import하여야 합니다. 자세한 내용은 다음 링크에 정리되어 있습니다. https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/api.html?highlight=optimier#module-tensorflow.python.ipu.optimizers

![스크린샷 2021-04-22 오후 5 06 19](https://user-images.githubusercontent.com/12121282/115678997-11d7be80-a38d-11eb-970e-a2174fc250a9.png)
* keras 모듈에서 정의된 모델을 IPU에서 사용할 수 있게 하기 위해 inject_ipukeras_modules함수로 efficientnet 모델을 wrapping합니다. 
* inject_ipukeras_modules 함수는 ```__init__.py```에 정의되어 있습니다.
```python
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
```
* IPU 버전의 모델은 input data로 일정한 batch size 단위로 묶어서 전달해야 합니다. 따라서 input_shape가 아닌 input_tensor로 변환하고 batch size를 설정합니다.
* IPU에서 double precision 연산은 지원하지 않기 때문에 데이터를 single precision으로 형변환합니다.

![스크린샷 2021-04-22 오후 5 42 00](https://user-images.githubusercontent.com/12121282/115684126-0cc93e00-a392-11eb-84d8-18a116b76e4e.png)
* CIFAR10 데이터셋을 다운받고, 데이터를 0~1 사이의 값으로 정규화하는 작업을 공통적으로 진행합니다.
* IPU 버전에서 데이터셋을 구성할 때는 위에서 언급한 batch size 단위로 데이터를 묶는 작업을 추가적으로 진행합니다.
* from_tensor_slices 함수로 데이터를 슬라이싱하고, repeat 함수는 데이터셋의 마지막에 도달했을 경우, 다시 처음부터 조회합니다. batch 함수 내부에 설정된 값만큼 batch의 개수를 지정하고, drop_remainder 옵션이 True로 설정되어 있으면 데이터셋을 모두 구성하고 남은 데이터를 제외시킵니다.

![스크린샷 2021-04-22 오후 5 54 57](https://user-images.githubusercontent.com/12121282/115686131-dbea0880-a393-11eb-8c78-d6453f51b0c2.png)
* main 함수에서는 GPU와 IPU에 대한 설정을 정의합니다.
* GPU 버전에서는 2개 이상의 GPU가 인식되었을 경우 자동으로 분산하여 학습을 진행하게 됩니다.
* ipu.utils.auto_select_ipus 함수는 학습시키는 데 사용할 IPU의 개수를 정의합니다. 모델이 __P__ 개의 IPU에 걸쳐 파이프라이닝되었다면, __P의 배수__ 로 값을 설정할 수 있습니다. 단, 그 값이 사용 가능한 IPU의 개수를 초과하면 안됩니다. 즉, 사용 가능한 IPU 개수를 N이라고 가정했을 때, (1 <= kP <= N, k는 자연수) 를 만족시켜야 합니다. 해당 조건이 만족되면 k-1개의 IPU에 자동으로 replication이 진행됩니다.
* 예를 들어, 사용 가능한 IPU의 개수가 32개이고, 4개의 IPU에 걸쳐 파이프라이닝된 모델에 auto_select_ipus 값을 24로 주게 되면, 총 24개의 IPU에서 4개의 파이프라이닝된 모델이 6번 복제되어 학습이 진행됩니다. 이를 replication이라고 하는데, replication이 된 만큼 더 많은 양의 batch를 처리하기 때문에 학습 효율을 높일 수 있습니다.

![스크린샷 2021-04-23 오전 10 09 38](https://user-images.githubusercontent.com/12121282/115803521-05e70d80-a41c-11eb-9385-90b83f2d3930.png)
* IPU에서는 학습을 진행할 모델을 포함한 전체적인 학습 과정을 with strategy.scope()라는 loop로 감싸주어야 합니다.
* 파이프라이닝된 모델을 불러오고, optimizer, 손실 함수를 정의하여 compile하고, 학습시킬 데이터셋을 불러옵니다.
* 학습을 중간에 모델을 저장하기 위한 체크포인트 경로를 설정합니다.

![스크린샷 2021-04-23 오전 10 44 09](https://user-images.githubusercontent.com/12121282/115805902-d981c000-a420-11eb-82c2-576df5bc0113.png)
* 학습을 진행할 때 사용할 세부적인 parameter를 설정합니다.
* steps_per_epoch는 한 epoch당 학습에 사용할 샘플의 갯수로, 보통 데이터 수를 배치 사이즈로 나눈 값으로 설정합니다. IPU 버전의 경우 repeat 함수를 통해 데이터셋을 구성하기 때문에 steps_per_epoch 값을 반드시 정의해주어야 합니다.

![스크린샷 2021-04-23 오후 1 38 51](https://user-images.githubusercontent.com/12121282/115819160-40ab6e80-a439-11eb-96f6-76b0854119a3.png)
* 학습시킨 모델을 test 데이터셋을 통해 평가하는 과정입니다.
* IPU 버전에서는 evaluate 함수 내부에서 steps라는 argument를 설정해주어야 합니다. steps_per_epoch와 유사한 값으로, 한 번에 몇개의 샘플을 평가할 지를 결정합니다. 일반적으로 steps_per_epoch와 같은 값으로 설정합니다.

### - efficientnet.keras의 EfficientNet 모델 vs efficientnet_model_ipu_param.py
![스크린샷 2021-04-23 오후 2 02 54](https://user-images.githubusercontent.com/12121282/115820833-9b929500-a43c-11eb-897a-a598818d4418.png)
* keras 모듈에서 지원하는 Efficientnet 모델과 IPU 환경에 맞춰 pipeline된 모델을 line by line으로 비교합니다.
* IPU 모델에서는 파이프라이닝을 하는 과정에서 IPU API를 사용하므로 ipu 모듈을 import하고, 사용하지 않는 모듈은 제거합니다.

![스크린샷 2021-04-23 오후 2 24 03](https://user-images.githubusercontent.com/12121282/115822481-9125ca80-a43f-11eb-80df-ea67d2223e0c.png)
* 모델의 각 block을 정의하는 매개변수의 집합인 BlockArgs는 namedtuple이라는 자료형으로 저장되어 있습니다.
* namedtuple은 튜플 내의 변수에 인덱스뿐만 아니라 이름으로도 접근할 수 있도록 설계되어 있는 자료구조입니다. ex) block_args.input_filters, block_args.se_ratio

![스크린샷 2021-04-23 오후 3 11 42](https://user-images.githubusercontent.com/12121282/115826665-393e9200-a446-11eb-8ae1-dfdb4a2caeaf.png)
* 파이프라이닝한 모델을 여러 IPU에 최대한 균일하게 분산시키기 위해 block_args를 담는 리스트를 선언하여 BlockArgs를 pipeline stage만큼 나누어 할당합니다.
* block_args_list에는 각 pipeline stage별 BlockArgs가 저장됩니다. ex) [1, 2, 3, 4, 5, 6, 7] => [[1, 2], [3, 4], [5, 6], [7]]

![스크린샷 2021-04-23 오후 3 19 46](https://user-images.githubusercontent.com/12121282/115827522-59bb1c00-a447-11eb-83d5-3edc0b19cc78.png)
* 배열 슬라이싱을 다르게 했기 때문에 그에 맞게 인덱싱을 추가해줍니다.

![스크린샷 2021-04-23 오후 6 02 52](https://user-images.githubusercontent.com/12121282/115847665-23d56200-a45e-11eb-8f2f-11b94d166dcd.png)
* IPU상에서 파이프라이닝 모델을 학습시키기 위해 ipu.keras.PipelineModel 함수로 wrapping해줍니다. gradient_accumulation_count 값은 __replication factor의 배수__ 로 설정해주어야 합니다.
* gradient_accumulation_count 값을 G, 배치 사이즈를 B라고 가정했을 때, B * G만큼 gradient를 저장해두었다가 한 번에 weight update를 진행한다는 의미입니다. 따라서 학습 과정에서 CPU와 IPU간의 latency를 줄여줍니다. 

## 학습 실행
```
python efficientnet_CIFAR10_train_ipu.py 
```
해당 파일이 존재하는 디렉토리에서 다음 커맨드를 실행하면 학습이 정상적으로 진행됩니다. 
