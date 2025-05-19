# Collaborative Scheduling for Active Particulate Air Pollution Reduction with Mobile Sprinklers


This repository contains the code for reproducing results in the thesis paper of Zijian Xiao.


# Getting Started

method one
1. Creating a Python virtual environment is recommended but not required. 

    ```bash
    conda env create -f sprinkle.yaml
    conda install --name sprinkle --file packages.txt
    conda activate sprinkle
    ```

2. Install Requirements
   
    ```bash
    pip install -r sprinkle.txt
    ```


## Reproducing The Results
This repository follows the following structure.

- `bat`: 批量运行的脚本文件.
- `experimentipy`: jupyter 文件 可一步步运行.
- `outputs`: 运行结果的输出位置.
- `pypolo2`: 搜索算法的主体部分.
    - `configs`: 脚本运行所需的配置文件.
    - `dynamics`: 暂时无用，考虑机器人的动力学的时候才需要.
    - `experiments`: 配合实验过程的功能函数.
    - `gridcontext`: 定义了搜索过程中的各类矩阵与内部计算.
    - `kernel`: 核函数.
    - `models`: 概率模型.
    - `objective`: 目标计算.
    - `robots`: 智能体实例.
    - `sensors`: 传感器实例.
    - `strategies`: 搜索算法.
    - `utilities`: 功能函数.
- `Sprayer_PDE`:基于物理过程的环境仿真
- `visual`: 根据实验结果画图.
- `main.py`: main file for running the experiments.

你可以运行bat文件夹下的脚本文件来运行，也可以用jupyter文件来一步步了解。

