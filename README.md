## introduction
hornbro is a tool for automated program repair using Homotopy-like Method for Automated Quantum Program Repair.
## installation
hornbro can be installed using conda and pip:
create a virtual environment with conda
```
conda create -n hornbroenv python==3.10
```
```
conda activate hornbroenv
```

install hornbro locally using pip

```
pip install -e .
```

## usage
hornbro can be used as a command line tool or as a python library.


## todo
1.	修复提交的代码和数据无法下载, 提供一个文档确定能复现 (R1 说不定要跑了复现)
2.	RB Q1中说怎么在SMT中描述测量概率分布
3.	Table 3和4 实验中增加Grover算法
4.	补一个实际位置和修复位置的实验，计算两者匹配的成功率，和修复所需的数量
5.	理一下实验中用的Assertion，讲下是怎么定义的，针对每个算法编一下
6.  跑一个full 和batch 的对比成功率
7.  多跑几个超参数的实验