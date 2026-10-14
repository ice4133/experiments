import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ==========================================
# 一、加载糖尿病数据集 diabetes (CSV文件)，观察数据
# ==========================================

# 1. 载入本地糖尿病数据库 CSV 文件，查看数据
# 【请将下方的 'diabetes.csv' 修改为你实际的 CSV 文件路径】
csv_path = 'homework/diabetes/data/load_diabetes.csv'  
raw_df = pd.read_csv(csv_path)

# 自动提取特征列名（默认除最后一列外的所有列为特征，最后一列为目标结果）
feature_names = list(raw_df.columns[:-1])
target_col_name = raw_df.columns[-1]

print("【步骤1】CSV 数据集加载成功！")
print("【步骤1】特征名称:", feature_names)
print("【步骤1】原始目標列名称:", target_col_name)
print("【步骤1】数据集整体形状 (行数, 列数):", raw_df.shape)

# 2. 切分数据，组合成 DataFrame 数据，并输出数据集前几行，观察数据
df = raw_df[feature_names].copy()

# 保留原始目标值（用于第 5 步散点图与相关性计算）
df['target_raw'] = raw_df[target_col_name]

# 智能判断：如果目标列本身已经是 0/1 二分类，则直接使用；如果是连续数值，则按中位数转为 0/1
if raw_df[target_col_name].nunique() <= 2:
    df['target'] = raw_df[target_col_name].astype(int)
    print("\n【步骤2】检测到目标列本身已是二分类标签 (0/1)，直接用于逻辑回归。")
else:
    median_val = np.median(raw_df[target_col_name])
    df['target'] = (raw_df[target_col_name] > median_val).astype(int)
    print(f"\n【步骤2】检测到目标列为连续值，已按中位数 ({median_val:.2f}) 转换为 0/1 二分类标签。")

print("\n【步骤2】输出组合后的 DataFrame 前 5 行数据:")
print(df.head())


# ==========================================
# 二、基于逻辑回归对数据集进行分析
# ==========================================

# 3. 查看数据集信息，从数据集中抽取训练集和测试集
print("\n【步骤3】查看数据集基本信息 (info):")
print(df.info())
print("\n【步骤3】查看数据集统计描述 (describe):")
print(df.describe())

# 提取特征 X 和二分类标签 y
X = df[feature_names]
y = df['target']

# 抽取训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 对特征进行标准化（消除 CSV 中不同列之间的量纲差异，保证逻辑回归正常收敛）
scaler_all = StandardScaler()
X_train_scaled = scaler_all.fit_transform(X_train)
X_test_scaled = scaler_all.transform(X_test)

print(f"\n【步骤3】数据切分完成 -> 训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# 4. 建立逻辑回归模型，训练数据，评估模型
model_all = LogisticRegression(max_iter=1000, random_state=42)
model_all.fit(X_train_scaled, y_train)

train_score = model_all.score(X_train_scaled, y_train)
test_score = model_all.score(X_test_scaled, y_test)
print(f"\n【步骤4】全特征逻辑回归 - 训练集准确率: {train_score:.4f}")
print(f"【步骤4】全特征逻辑回归 - 测试集准确率: {test_score:.4f}")


# ==========================================
# 三、考察每个特征值与结果之间的关联性，观察得出最相关的特征
# ==========================================

# 5. 考察每个特征值与结果之间的关系，分别以散点图展示
num_features = len(feature_names)
cols = 4  # 每行画 4 个子图
rows = int(np.ceil(num_features / cols))

plt.figure(figsize=(4.5 * cols, 3.8 * rows))
for i, col in enumerate(feature_names):
    plt.subplot(rows, cols, i + 1)
    plt.scatter(df[col], df['target_raw'], c=df['target'], cmap='coolwarm', alpha=0.6, edgecolors='k', s=25)
    corr = df[col].corr(df['target_raw'])
    plt.title(f"{col} (corr = {corr:.2f})")
    plt.xlabel(col)
    plt.ylabel(target_col_name)

plt.tight_layout()
plt.show()

# 思考：根据散点图结果对比，哪个特征值与结果之间的相关性最高？
correlations = df[feature_names].apply(lambda x: x.corr(df['target_raw'])).abs().sort_values(ascending=False)
best_feature = correlations.index[0]  # 自动获取相关性排名第一的特征名称

print("\n【步骤5 - 思考题回答】各特征与结果的相关系数绝对值排序:")
print(correlations)
print(f"\n>>> 结论：根据散点图与相关系数对比，【{best_feature}】特征与结果之间的相关性最高（相关系数为 {correlations.iloc[0]:.4f}）！")


# ==========================================
# 四、使用回归分析找出最相关特征值与糖尿病的关联性，并预测出相关结果
# ==========================================

# 6. 把 5 中相关性最高的特征值提取，然后进行数据切分
X_single = df[[best_feature]]  # 保持二维 DataFrame 结构
y_single = df['target']

X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
    X_single, y_single, test_size=0.2, random_state=42, stratify=y_single
)
print(f"\n【步骤6】已提取最相关特征【{best_feature}】并完成切分 -> 训练集: {X_s_train.shape}, 测试集: {X_s_test.shape}")

# （注：原题目序号跳过 7，直接对应 8）
# 8. 创建逻辑回归模型，进行模型训练
model_single = LogisticRegression(random_state=42)
model_single.fit(X_s_train, y_s_train)
print(f"【步骤8】基于单特征【{best_feature}】的逻辑回归模型训练完成。")

# 9. 对测试集进行预测，求出权重系数
y_s_pred = model_single.predict(X_s_test)
y_s_prob = model_single.predict_proba(X_s_test)[:, 1]

w_val = model_single.coef_[0][0]
b_val = model_single.intercept_[0]

print(f"\n【步骤9】测试集前 10 个样本的预测结果 (0/1):", y_s_pred[:10])
print(f"【步骤9】求出的权重系数 (w / coef_): {w_val:.4f}")
print(f"【步骤9】求出的偏置截距 (b / intercept_): {b_val:.4f}")

# 10. 对预测结果进行评价，结果可视化
acc_single = accuracy_score(y_s_test, y_s_pred)
auc_single = roc_auc_score(y_s_test, y_s_prob)
print(f"\n【步骤10】单特征模型评价 - 测试集准确率 (Accuracy): {acc_single:.4f}")
print(f"【步骤10】单特征模型评价 - ROC-AUC 得分: {auc_single:.4f}")
print("\n【步骤10】详细分类评估报告:\n", classification_report(y_s_test, y_s_pred))

# 结果可视化：绘制测试集散点 + Sigmoid 概率曲线 + 0.5 决策边界
plt.figure(figsize=(9, 5))

# 1) 绘制测试集的真实样本点
plt.scatter(X_s_test[best_feature], y_s_test, color='steelblue', alpha=0.6, label='Actual Labels (Test Set)')

# 2) 生成平滑曲线数据，绘制逻辑回归 Sigmoid 概率曲线
x_min, x_max = X_single[best_feature].min(), X_single[best_feature].max()
margin = (x_max - x_min) * 0.1
x_range = np.linspace(x_min - margin, x_max + margin, 300).reshape(-1, 1)
y_range_prob = model_single.predict_proba(pd.DataFrame(x_range, columns=[best_feature]))[:, 1]

plt.plot(x_range, y_range_prob, color='crimson', linewidth=2.5, label='Logistic Regression Sigmoid Curve')

# 3) 画出 0.5 概率阈值线与垂直决策边界 (wx + b = 0  =>  x = -b / w)
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Probability Threshold = 0.5')
if w_val != 0:
    decision_boundary = -b_val / w_val
    plt.axvline(decision_boundary, color='forestgreen', linestyle='--', alpha=0.8, 
                label=f'Decision Boundary ({best_feature} = {decision_boundary:.2f})')

plt.title(f"Step 10: Logistic Regression on '{best_feature}' vs. Diabetes")
plt.xlabel(best_feature)
plt.ylabel("Probability of Diabetes (Target = 1)")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()