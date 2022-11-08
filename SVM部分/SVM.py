import scipy.io as sio
from libsvm.svmutil import *
from libsvm.svm import *

# 将输入数据转换为libsvm需要的数据格式
def get_train_data():
    train_data = sio.loadmat('./train(Task_2)/train_data.mat')['train_data']
    train_label = sio.loadmat('./train(Task_2)/train_label.mat')['train_label']

    print('shape of train_data: ', train_data.shape)
    with open('train.txt', 'w') as f:
        for i in range(train_data.shape[0]):
            label = train_label[i]
            f.write(str(label.item()))
            for j in range(train_data.shape[1]):
                f.write(f' {j}:{train_data[i,j]}')
            f.write('\n')

def get_test_txt():
    test_data = sio.loadmat('./test(Task_2)/test_data.mat')['test_data']

    print('shape of train_data: ', test_data.shape)
    with open('train.txt', 'w') as f:
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                f.write(f' {j}:{test_data[i, j]}')
            f.write('\n')
# 将测试数据转换为libsvm需要的数据格式
def get_test_data():
    test_data = sio.loadmat('./test(Task_2)/test_data.mat')['test_data']
    result = []
    for i in range(test_data.shape[0]):
        dic = {}
        for j in range(test_data.shape[1]):
            dic1 = {j:test_data[i,j]}
            dic.update(dic1)
        result.append(dic)
    return result


def get_test_label():
    import numpy as np
    ground_label = []
    f = open("test_label.txt", encoding="utf-8")
    for line in f.readlines():
        line_label = int(line.strip().split(" ")[-1])
        ground_label.append(line_label)
    return np.array(ground_label)
get_train_data()
test_data = get_test_data()

y, x = svm_read_problem('train.txt')
prob = svm_problem(y, x)

param = svm_parameter('-t 2 -c 0.11 -b 1 -d 2 -g 0.105')
model = svm_train(prob, param)
# model = svm_load_model('out.model')
print("train done!")
svm_save_model('svm_model_file.model', model)

print('test:')
test_label = get_test_label()
p_label, p_acc, p_val = svm_predict(test_label, test_data, model)
print(p_label)
print('predict done!')
#将预测结果写入txt文件
with open('result_svm.txt', 'w') as f:
    for i in range(len(p_label)):
        f.write(f'{i+1} {int(p_label[i])}\n')
