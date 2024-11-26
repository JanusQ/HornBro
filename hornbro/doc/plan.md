# 量子程序修复 (Robustness)
<!-- RUNNER: Responsible UNfair NEuron Repair for Enhancing Deep Neural Network Fairness -->

## 目标
1. 验证
2. 定位
3. 修复

## 挑战
1. 验证：空间大, profiling 的 代价大 
2. 定位：没有办法获得中间的状态，难以验证是否是应该修复的位置
3. 修复：难以找到效率高的电路模块

### 方法（guarantee）
1. 分析手段：
    基于切割的近似，不提密度矩阵，只提概率分布
    概率分布下的优化空间
    overhead optimization (特征分解，Clifford 近似输入)
2. 直接修复：
    优化目标：Sin 下电路的 Assert 是对
    换成参数化门，梯度下降（和正确采样输出的距离），写下 parameter shift 计算公式
3. 语义保持的修复（最小化变化）
    Clifford Local Search 搜索结构（怎么搜索）
    定义变化空间（编辑距离）位移，添加，SWAP
    计算 Clifford, 和近似输入下的平均距离（允许提前结束）

<!-- ，换比特的操作，mirror -->

    importance estimation
        1. gradient-based 

    Unitary 替换

    Clifford 近似
    Local Search


2. 定位：
    核心：, parameter shift，计算 graident 的 绝对值的平均 （不确定是否完全正确）
    问题：如何保证关键的部分被发现 
    
3. 修复：代价小，不会改变原先是正确的输出
    Clifford 近似 -> 变成参数化电路搜索


    插入逆向，保证最差情况下是 identity 的


# 实验的数据集
软工常用的 Bug4G，编译测试生成的