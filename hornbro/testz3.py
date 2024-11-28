from z3 import *

# 创建 Z3 求解器
solver = Solver()
solver.set("logic", "QF_BV")
# 假设有5个布尔变量，表示为 1 位的位向量
gatelist = [BitVec(f"gate_{i}", 1) for i in range(5)]

# 添加约束：恰好有一个布尔变量为 True
solver.add(Sum(gatelist) == 1)

# 检查是否有解
if solver.check() == sat:
    model = solver.model()
    # 输出满足条件的模型
    print([model[g] for g in gatelist])
else:
    print("Unsatisfiable")
