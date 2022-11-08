# 1. 导入库
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs, make_circles
import matplotlib
import imageio.v2 as imageio

# 制作动图函数
def create_gif(image_list, gif_name, duration):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


matplotlib.rcParams['font.sans-serif'] = ['FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False
# 2.创建分割数据
X, y = make_blobs(n_samples=150, centers=2, cluster_std=[5, 3], random_state=504)
X, y = make_circles(n_samples=70, noise=0.2, factor=0.2, random_state=1)
X = X * 4
# 可视化数
colorname = ['r', 'b']
plt.figure(1)
plt.xlim(-5, 5)
plt.ylim(-6, 8)
for i in range(len(y)):
    plt.scatter(X[i, 0], X[i, 1], c=colorname[y[i]], marker='o', s=25)
plt.title('线性不可分数据集')

ax = plt.gca()  # 获取当前子图,如果不存在, 则创建新的子图

# 制作网格
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# 在最大值和最小值之间形成30个规律的数据
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
# 使用meshgrid()将两个一维向量转为特征矩阵, 以便获取y.shape*x.shape这么多个坐标点的横纵坐标
XX, YY = np.meshgrid(xx, yy)
# 形成网格,vstack()将多个结构一致的一维数组按行堆叠起来
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# plt.scatter(xy[:, 0], xy[:, 1], s=1, cmap='rainbow')

# 惩罚因子c
C = [0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50]

# 动图图片列表
image_list = []
xx = 5
yy = 8
for n in range(len(C)):
    c = C[n]
    # 拟合模型
    clf = svm.SVC(kernel='rbf', C=C[n], gamma=0.1)
    clf.fit(X, y)
    plt.figure(2)
    plt.xlim(-5, 5)
    plt.ylim(-6, 8)
    for i in range(len(y)):
        plt.scatter(X[i, 0], X[i, 1], c=colorname[y[i]], marker='o', s=25)
    # decision_function, 返回每个输入样本对应的决策边界的距离
    ax = plt.gca()  # 获取当前子图
    ax.set_title('线性不可分')
    Z = clf.decision_function(xy)
    # print('Z:{}'.format(Z.shape))
    # contour要求Z的结构需要与X和Y一致
    Z = Z.reshape(XX.shape)
    # print('new Z:{}'.format(Z.shape))
    # 绘制决策边界和边际
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

    # 绘制支持向量
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
               edgecolors='k')

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
    # 探索建好的模型
    clf.predict(X)  # 根据决策边界, 对X中的样本进行分类,返回的结构为n_samples
    print(clf.score(X, y))  # 返回给定测试数据和target的平均准确度
    # 保存生成的图片
    if os.path.exists('SVMplot_output') == 0:
        os.mkdir('SVMplot_output')
    plt.savefig('SVMplot_output' + '//线性不可分2_C' + str(n) + '.jpg', dpi=100)
    image_list.append('SVMplot_output' + '//线性不可分2_C' + str(n) + '.jpg')
    if n == (len(C) - 1):
        image_list.append('SVMplot_output' + '//线性不可分2_C' + str(n) + '.jpg')
        image_list.append('SVMplot_output' + '//线性不可分2_C' + str(n) + '.jpg')

    plt.show()

# 制作动图
gif_name = 'SVMplot_output' + '//线性不可分2.gif'
duration = 0.3  # 单位s
create_gif(image_list, gif_name, duration)
