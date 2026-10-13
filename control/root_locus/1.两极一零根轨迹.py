# 构造一个两极一零 根轨迹， 圆弧
import control as ctrl
import matplotlib.pyplot as plt


# 1. 定义开环系统模型： G(s) = K * (s + z) / (s^2 + 2ζω_n s + ω_n^2)
# 同时学习， 如何构造一个任意项传递函数
# tf函数，tf(num, den) 传入的列表本质上是从高次幂到低次幂的多项式系数
num = [1, 5]  # 分子多项式系数
den = [1, 4, 3]  # 分母多项式系
sys = ctrl.tf(num, den)  # 构造传递函数

# 2. 根轨迹计算与空间网格
plt.figure(figsize=(7, 6))
ctrl.root_locus(sys, grid=True)

# 3. 复平面基准线与几何标注
plt.title("Root Locus Plot of G(s) = K / [s(s + 2)]")
plt.xlabel("Real Axis (seconds⁻¹)")
plt.ylabel("Imaginary Axis (seconds⁻¹)")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--') # 实轴 (σ 轴)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--') # 虚轴 (jω 轴)

# 4. 图像固化与显示
plt.savefig("root_locus.png", dpi=300, bbox_inches='tight')
plt.show()