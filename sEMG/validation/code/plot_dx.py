import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# 初始化模型列表
models = [load(f'dx_model_subject{i}_Random Forest.joblib') for i in range(1, 7)]  # 加载9个模型

# 初始化存储所有样本结果的字典
all_Movements = {f'Week {i-1}': {j: [] for j in range(10, 16)} for i in range(1, 4)}  # 3个样本，每个样本9个动作

# 遍历三个样本数据
for i in range(1, 4):
    new_data_path = fr'D:\SEMG\data\prodata\ML6\7-{i}d.mat'
    new_data = scio.loadmat(new_data_path)
    X = new_data['data']
    y_new = new_data['label'].ravel() if 'label' in new_data else None  # 可选真实标签

    # 打印数据信息
    print(f"样本 {i} 数据形状:", X.shape)
    print(f"样本 {i} 标签是否存在:", 'label' in new_data)

    # 对每个模型进行预测
    y_preds = [model.predict(X) for model in models]  # 9个模型的预测结果

    # 根据标签将预测结果分组
    for k in range(len(X)):
        label_str = y_new[k][2].strip()  # 去除首尾空格
        if label_str.isdigit():
            label = int(label_str)
            if 10 <= label <= 15:
                all_Movements[f'Week {i-1}'][label].append(y_preds[label-10][k])  # 将对应模型的预测结果加入

# 将分组结果转换为 DataFrame
data = []
for Week, Movements in all_Movements.items():
    for Movement, preds in Movements.items():
        for pred in preds:
            data.append({'Week': Week, 'Movement': f'Movement{Movement}', 'Prediction': pred})

df = pd.DataFrame(data)

# 设置全局字体
# plt.rcParams['font.family'] = 'Times New Roman'

# # 创建图形和主坐标轴
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # 绘制分箱散点图（主坐标轴）
# sns.stripplot(
#     x='Movement',  # 横坐标为动作类别
#     y='Prediction',  # 纵坐标为预测结果
#     hue='Week',  # 按样本分组着色
#     data=df,  # 数据源
#     palette='Set1',  # 配色方案
#     jitter=True,  # 添加抖动避免点重叠
#     dodge=True,  # 按样本分组分箱
#     size=6,  # 点的大小
#     alpha=0.7,  # 点的透明度
#     linewidth=0.1,  # 点边缘线宽
#     edgecolor='black',  # 点边缘颜色
#     ax=ax1  # 指定主坐标轴
# )

# # 设置主坐标轴的标题和标签
# # ax1.set_title('Patient\'s Three-week FMA-UE Score of 6 Forearm Pronation-supination Movements', fontsize=18, fontweight='bold', pad=20)
# # ax1.set_xlabel('Movement', fontsize=14, fontweight='bold')
# ax1.tick_params(axis='x', pad=2)  # X轴刻度标签间距
# ax1.tick_params(axis='y', pad=2)  # Y轴刻度标签间距
# ax1.set_ylabel('Predicted value of patient\'s FMA-UE', fontsize=14, fontweight='bold')
# ax1.set_yticks([0, 1, 2])  # 设置纵坐标刻度
# ax1.set_ylim(-0.2, 2.1)
# ax1.set_xlabel('')
# ax1.legend(title='Predicted Value', 
#     handlelength=1,    # 控制符号框长度
#     handleheight=1,    # 控制符号框高度
#     handletextpad=0.1,   # 符号与文本间距
#     borderpad=0.2,       # 图例内边距
#     columnspacing=0.2,   # 列间空白
#     fontsize=8,
#     title_fontsize=10,
#     frameon=True,
#     bbox_to_anchor=(0.9, 0.75),
#     loc='upper left')  # 添加图例

# # 创建副坐标轴
# ax2 = ax1.twinx()

# # 计算每个 Movement 的柱状图数据
# Movement_list = [f'Movement{i}' for i in range(1, 7)]
# bar_values = [0.01 if week == 'Week 0' else 1 for week in df['Week']]  # Week 0 为 0，Week 1 和 Week 2 为 1

# # 绘制柱状图（副坐标轴）
# sns.barplot(
#     x='Movement',  # 横坐标为动作类别
#     y=bar_values,  # 纵坐标为柱状图的值
#     hue='Week',  # 按样本分组着色
#     data=df,  # 数据源
#     palette='Set1',  # 配色方案
#     alpha=0.3,  # 柱状图透明度
#     ax=ax2  # 指定副坐标轴
# )

# # 设置副坐标轴的标签
# ax2.tick_params(axis='x', pad=2)  # X轴刻度标签间距
# ax2.tick_params(axis='y', pad=2)  # Y轴刻度标签间距
# ax2.set_ylabel('Real value of patient\'s FMA-UE', fontsize=14, fontweight='bold')
# ax2.set_yticks([0, 1, 2])  # 设置纵坐标刻度
# ax2.set_ylim(-0.2, 2.1)  # 设置纵坐标范围，与主坐标轴对齐
# ax2.set_xlabel('')
# ax2.legend(title='    Real Value    ', 
#     handlelength=1,    # 控制符号框长度
#     handleheight=1,    # 控制符号框高度
#     handletextpad=0.1,   # 符号与文本间距
#     borderpad=0.2,       # 图例内边距
#     columnspacing=0.2,   # 列间空白
#     fontsize=8,
#     title_fontsize=10,
#     frameon=True,
#     bbox_to_anchor=(0.9, 0.75),
#     loc='upper right')  # 添加图例

# # 调整布局
# plt.tight_layout()

# # 显示图形
# plt.show()

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'

# 创建图形和主坐标轴
fig, ax1 = plt.subplots(figsize=(14, 6))

# 绘制分箱散点图（主坐标轴）
sns.stripplot(
    x='Movement',  # 横坐标为动作类别
    y='Prediction',  # 纵坐标为预测结果
    hue='Week',  # 按样本分组着色
    data=df,  # 数据源
    palette='Set1',  # 配色方案
    jitter=0.25,  # 添加抖动避免点重叠
    dodge=True,  # 按样本分组分箱
    size=6,  # 点的大小
    alpha=0.7,  # 点的透明度
    linewidth=0.1,  # 点边缘线宽
    edgecolor='black',  # 点边缘颜色
    ax=ax1  # 指定主坐标轴
)

# 设置主坐标轴的标题和标签
# ax1.set_title('Patient\'s Three-week FMA-UE Score of 9 Elbow Flexion-extension Movements', fontsize=18, fontweight='bold', pad=20)
# ax1.set_xlabel('Movement', fontsize=14, fontweight='bold')
# ax1.tick_params(axis='x', pad=2)  # X轴刻度标签间距
# ax1.tick_params(axis='y', pad=2)  # Y轴刻度标签间距
ax1.set_ylabel('Model predicted value of \n patient\'s FMA-UE', fontsize=14, fontweight='bold')
ax1.set_yticks([0, 1, 2])  # 设置纵坐标刻度
ax1.set_ylim(-0.2, 2.1)
ax1.set_xlabel('')
ax1.legend(title='   Model Predicted Value   ', 
    # handlelength=1,    # 控制符号框长度
    # handleheight=1,    # 控制符号框高度
    # handletextpad=0.1,   # 符号与文本间距
    # borderpad=0.2,       # 图例内边距
    # columnspacing=0.2,   # 列间空白
    fontsize=8,
    title_fontsize=10,
    frameon=True,
    bbox_to_anchor=(0.85, 0.75),
    loc='center')  # 添加图例

# 创建副坐标轴
ax2 = ax1.twinx()

# 计算每个 Movement 的柱状图数据
Movement_list = [f'Movement{i}' for i in range(1, 7)]
bar_values = [0.01 if week == 'Week 0' else 1 for week in df['Week']]  # Week 0 为 0，Week 1 和 Week 2 为 1

# 绘制柱状图（副坐标轴）
sns.barplot(
    x='Movement',  # 横坐标为动作类别
    y=bar_values,  # 纵坐标为柱状图的值
    hue='Week',  # 按样本分组着色
    data=df,  # 数据源
    palette='Set1',  # 配色方案
    alpha=0.3,  # 柱状图透明度
    ax=ax2  # 指定副坐标轴
)

# 设置副坐标轴的标签
# ax2.tick_params(axis='x', pad=2)  # X轴刻度标签间距
# ax2.tick_params(axis='y', pad=2)  # Y轴刻度标签间距
ax2.set_xlabel('')
ax2.set_ylabel('Clinical assessment of \n patient\'s FMA-UE', fontsize=14, fontweight='bold')
ax2.set_yticks([0, 1, 2])  # 设置纵坐标刻度
ax2.set_ylim(-0.2, 2.1)  # 设置纵坐标范围，与主坐标轴对齐
ax2.grid(  # 新增网格
    True,
    axis='y',
    linestyle='--',
    alpha=0.5,
    color='orange',
    zorder=0
)
ax2.legend(title='Clinical Assessment Value', 
    # handlelength=1,    # 控制符号框长度
    # handleheight=1,    # 控制符号框高度
    # handletextpad=0.1,   # 符号与文本间距
    # borderpad=0.2,       # 图例内边距
    # columnspacing=0.2,   # 列间空白
    fontsize=8,
    title_fontsize=10,
    frameon=True,
    bbox_to_anchor=(0.7, 0.75),
    loc='right')  # 添加图例

# 调整布局
plt.tight_layout(pad=0.4)
plt.subplots_adjust(left=0.1, right=0.9,top=0.85, bottom=0.15)
# 显示图形
plt.show()