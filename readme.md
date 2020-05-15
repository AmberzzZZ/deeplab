### 复现deeplabV3论文中的最强版本：
    resnet50, stride8, MG+ASPP

### 自然图像语义分割
    在医学图像上结果不会更好，且参数量大

### 改进方向
    参考：https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
    1. backbone：Xception, MobileNet
    2. ResNet starting stage 的7x7 替换成 3个3x3
    3. encoder-decorder结构：deeplabV3+


### 参数量
|  model  |  params  |
|  -----  |  ------  |
| ResNet50  | 1,460,096 |
| Xception  | 63,268,952 |
| deeplabV3  | 12,482,645 |
| deeplabV3+  | 80,119,389 |