# 说明
本项目是在[flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) 上改写成的，为了方便在gpu上进行光流提取。如有疑问，可以查看原文档。
## 安装配置
#### 1.安装依赖包
* numpy 
* PyTorch (本人采用1.1.0版本)
* scipy 
* scikit-image
* tensorboardX
* colorama, tqdm, setproctitle
#### 2.运行配置脚本
`bash install.sh`

1. 请注意，如果采用了anaconda环境，有可能要修改install.sh。要在bash脚本中activate 相应的环境。把~/.bashrc中，与anaconda相关的内容粘贴到install.sh开头(在#!/bin/bash之后)。
2. 如果有C++ 11相关的报错，可以尝试修改channelnorm_package, correlation_package, 和 resample2d_package 中的setup.py文件。

#### 3.下载预训练模型
* [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[620MB]
* [FlowNet2-C](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)[149MB]
* [FlowNet2-CS](https://drive.google.com/file/d/1iBJ1_o7PloaINpa8m7u_7TsLCX0Dt_jS/view?usp=sharing)[297MB]
* [FlowNet2-CSS](https://drive.google.com/file/d/157zuzVf4YMN6ABAQgZc8rRmR5cgWzSu8/view?usp=sharing)[445MB]
* [FlowNet2-CSS-ft-sd](https://drive.google.com/file/d/1R5xafCIzJCXc8ia4TGfC65irmTNiMg6u/view?usp=sharing)[445MB]
* [FlowNet2-S](https://drive.google.com/file/d/1V61dZjFomwlynwlYklJHC-TLfdFom3Lg/view?usp=sharing)[148MB]
* [FlowNet2-SD](https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view?usp=sharing)[173MB]

在项目的根目录下创建`checkpoints`文件夹，将以上预训练模型放在该文件夹下。

## 使用
修改extract_flow.sh脚本(具体可查看该脚本中的说明)
1. 将有关conda的配置信息粘贴到该脚本开头
2. 按个人需求修改 `dataset_name`, `video_dir`, `ext`, `thresh_hold`, `model_name`这五个变量。

运行 `bash extract_flow.sh`命令来进行提取光流特征，以及查看提取结果。

在运行时如果碰到Segmentation fault (core dumped)错误，
请在utils.flow_utils中的import matplotlib后添加 `matplotlib.use('TKAgg')`。
如果碰到的时关于`matplotlib.use('TKAgg')`的报错，就请注释这一行。


## Results
[![Predicted flows on MPI-Sintel](./image.png)](https://www.youtube.com/watch?v=HtBmabY8aeU "Predicted flows on MPI-Sintel")

