import os
from libsvm.svmutil import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import imageio.v2 as imageio


# 制作动图函数
def create_gif(image_list, gif_name, duration=1.0):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


# 设置随机种子和字体
np.random.seed(1)
matplotlib.rcParams['font.sans-serif'] = ['FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# 选择线性可分或者线性不可分数据集
Linearly_separable = True
# Linearly_separable = False

# 随机创建数据集
'''线性可分数据集'''
lx1 = -4 + 3.5 * np.random.rand(30)
lx2 = 4 - 3.5 * np.random.rand(30)
ly = -4 + 8 * np.random.rand(60)

plt.figure(1)
plt.xlim(-5, 5)
plt.ylim(-6, 8)
for i in range(len(lx1)):
    s1 = plt.scatter(lx1[i], ly[i], marker='o', s=25, c='r')
    s2 = plt.scatter(lx2[i], ly[i + 30], marker='o', s=25, c='b')
plt.legend((s1, s2), ('第一类样本', '第二类样本'), loc='best')
plt.title('线性可分数据集')

'''线性不可分数据集'''
ux1 = -4 + 4.5 * np.random.rand(30)
ux2 = 4 - 4.5 * np.random.rand(30)
uy = -4 + 8 * np.random.rand(60)

plt.figure(2)
plt.xlim(-5, 5)
plt.ylim(-6, 8)
for i in range(len(ux1)):
    s1 = plt.scatter(ux1[i], uy[i], marker='o', s=25, c='r')
    s2 = plt.scatter(ux2[i], uy[i + 30], marker='o', s=25, c='b')
plt.legend((s1, s2), ('第一类样本', '第二类样本'), loc='best')
plt.title('线性不可分数据集')

# 转换数据格式为SVM标准格式
with open('linear.txt', 'w') as f:
    for i in range(len(ly)):
        label = -1 if i < 30 else 1
        f.write(str(label))
        if i < 30:
            f.write(f' 1:{lx1[i]}')
            f.write(f' 2:{ly[i]}')
        else:
            f.write(f' 1:{lx2[i - 30]}')
            f.write(f' 2:{ly[i]}')
        f.write('\n')

with open('unlinear.txt', 'w') as f:
    for i in range(len(uy)):
        label = -1 if i < 30 else 1
        f.write(str(label))
        if i < 30:
            f.write(f' 1:{ux1[i]}')
            f.write(f' 2:{uy[i]}')
        else:
            f.write(f' 1:{ux2[i - 30]}')
            f.write(f' 2:{uy[i]}')
        f.write('\n')

data_type = ''
x1 = []
y1 = []

# 选择线性可分数据，和线性不可分数据
if Linearly_separable:
    X1 = [[0, 0] for i in range(60)]
    for i in range(30):
        X1[i][0] = lx1[i]
        X1[i + 30][0] = lx2[i]
        X1[i][1] = ly[i]
        X1[i + 30][1] = ly[i + 30]
    Y1 = [0] * 30 + [1] * 30
    X1 = np.array(X1)
    Y1 = np.array(Y1)
    data_type = '线性可分'
    # 读数据
    y1, x1 = svm_read_problem('linear.txt')
else:
    X1 = [[0, 0] for i in range(60)]
    for i in range(30):
        X1[i][0] = ux1[i]
        X1[i + 30][0] = ux2[i]
        X1[i][1] = uy[i]
        X1[i + 30][1] = uy[i + 30]
    Y1 = [0] * 30 + [1] * 30
    X1 = np.array(X1)
    Y1 = np.array(Y1)
    data_type = '线性不可分'
    y1, x1 = svm_read_problem('unlinear.txt')

# 惩罚因子c
C = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5]

# 动图图片列表
image_list = []

for n in range(len(C)):
    c = C[n]
    # 训练，选线性模型
    model1 = svm_train(y1, x1, '-t 0 -c ' + str(c))
    print(model1)
    sv_list = []

    # 创建绘图窗口
    plt.figure(3)
    ax = plt.gca()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(data_type)

    sv_x1 = []
    sv_y1 = []
    # 找支持向量SV的坐标
    sv1 = model1.get_SV()
    # 找支持向量SV在训练集中的序号，从1开始，注意与列表序号区别，需要减一，y2[sv_idx[i]-1]
    sv_idx = model1.get_sv_indices()
    print(sv_idx)

    for i in range(len(sv_idx)):
        sv_y1.append(int(y1[sv_idx[i] - 1]))  # 通过序号找到标签
        for j in range(len(sv1[0])):
            sv_list.append(sv1[i][j + 1])  # 特征序号从1，开始，故需j + 1
        sv_x1.append(sv_list)
        sv_list = []
    sv_x1 = np.array(sv_x1).reshape(i + 1, j + 1)

    # 系数alpha * y
    alpha1 = model1.get_sv_coef()

    # 由alpha和支持向量坐标求决策面系数w，画支持向量
    w = 0.0
    for i in range(len(alpha1)):
        w = w + alpha1[i][0] * sv_x1[i]  # 求w的公式
        # 画支持向量散点图
        ax.scatter(sv_x1[i, 0], sv_x1[i, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
        # if abs(alpha1[i][0]) < C[n]:
        #     if sv_y1[i] == -1:
        #         ax.scatter(sv_x1[i, 0], sv_x1[i, 1], marker='*', color="red", linewidths=1, s=150)
        #     else:
        #         ax.scatter(sv_x1[i, 0], sv_x1[i, 1], marker='*', color="blue", linewidths=1, s=150)
        # else:
        #     if sv_y1[i] == -1:
        #         ax.scatter(sv_x1[i, 0], sv_x1[i, 1], marker="^", color="red", linewidths=1, s=110)
        #     else:
        #         ax.scatter(sv_x1[i, 0], sv_x1[i, 1], marker="^", color="blue", linewidths=1, s=110)

    # 决策面偏置b
    b = model1.rho[0]
    print(b)
    # 画数据散点图
    ax.scatter(X1[0:30, 0], X1[0:30, 1], marker='o', s=25, color='r')
    ax.scatter(X1[30:60, 0], X1[30:60, 1], marker='o', s=25, color='b')

    # 划分界面和边缘分界面
    xx = 5
    yy = 8
    plt.xlim(-5, 5)
    plt.ylim(-6, 8)
    dis_x = np.arange(-xx, xx, 0.1)
    dis_y = np.arange(-yy, yy, 0.1)
    dis_x, dis_y = np.meshgrid(dis_x, dis_y)
    z = w[0] * dis_x + w[1] * dis_y - b
    ax.contour(dis_x, dis_y, z, [-1, 0, 1], colors='k', linestyles=['--', '-', '--'])

    # 图例边框
    ax.plot([1.8, 1.8], [8, 4.2], c='k', linewidth=0.5)
    ax.plot([1.8, 5], [4.2, 4.2], c='k', linewidth=0.5)
    # 显示惩罚项系数
    ax.text(xx - 3.2, yy - 0.7, ' C = ' + str(c), fontdict={'size': '11', 'color': 'k'})
    # 显示样本点类型
    ax.scatter(xx - 2.9, yy - 1.2, marker='o', s=25, color='r')
    ax.text(xx - 2.75, yy - 1.4, "第一类", fontdict={'size': '11', 'color': 'k'})
    ax.scatter(xx - 1.6, yy - 1.2, marker='o', s=25, color='b')
    ax.text(xx - 1.4, yy - 1.4, "第二类", fontdict={'size': '11', 'color': 'k'})
    # 显示分界面类型
    ax.plot([2, 2.7], [6, 6], c='k', linestyle='-')
    ax.text(xx - 2, yy - 2.15, "分界面", fontdict={'size': '11', 'color': 'k'})
    ax.plot([2, 2.7], [5.25, 5.25], c='k', linestyle='--')
    ax.text(xx - 2, yy - 2.9, "边缘分界面", fontdict={'size': '11', 'color': 'k'})
    # 显示支持向量
    ax.scatter(xx - 2.65, yy - 3.45, s=100, linewidth=1, facecolors='none', edgecolors='k')
    ax.text(xx - 2, yy - 3.65, "支持向量", fontdict={'size': '11', 'color': 'k'})
    # ax.scatter(xx - 2.9, yy - 3.5, marker='^', color="red", s=80)
    # ax.scatter(xx - 2.4, yy - 3.5, marker='^', color="blue", s=80)
    # ax.text(xx - 2, yy - 3.75, "其他支持向量", fontdict={'size': '11', 'color': 'k'})
    # 保存生成的图片
    if os.path.exists('SVMplot_output') == 0:
        os.mkdir('SVMplot_output')
    plt.savefig('SVMplot_output' + '//' + data_type + '_C' + str(n) + '.jpg', dpi=100)
    image_list.append('SVMplot_output' + '//' + data_type + '_C' + str(n) + '.jpg')
    if n == len(C) - 1:
        image_list.append('SVMplot_output' + '//' + data_type + '_C' + str(n) + '.jpg')
        image_list.append('SVMplot_output' + '//' + data_type + '_C' + str(n) + '.jpg')
    plt.show()

# 制作动图
gif_name = 'SVMplot_output' + '//' + data_type + '.gif'
duration = 0.3  # 单位s
create_gif(image_list, gif_name, duration)
