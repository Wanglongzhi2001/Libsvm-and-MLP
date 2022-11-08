import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR  # 动态修改学习率
import numpy as np
from tqdm import tqdm  # 加载进度条的库
import scipy.io as sio
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from random import shuffle


# 定义自己的数据集
class My_dataset(Dataset):
    def __init__(self, data, label, mode, transform=None):
        self.transform = transform
        self.mode = mode
        self.real_len = 0
        if mode == 'train':
            self.train_data = data[:700]
            self.train_label = label[:700]
            self.real_len = len(self.train_label)
        elif mode == 'valid':
            self.valid_data = data[700:]
            self.valid_label = label[700:]
            self.real_len = len(self.valid_label)
        elif mode == 'train_only':
            self.train_only_data = data
            self.train_only_label = label
            self.real_len = len(self.train_only_label)

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self.train_data[idx].reshape(19, 19)
            label = self.train_label[idx]
        elif self.mode == 'valid':
            data = self.valid_data[idx].reshape(19, 19)
            label = self.valid_label[idx]
        elif self.mode == 'train_only':
            data = self.train_only_data[idx].reshape(19, 19)
            label = self.train_only_label[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return self.real_len


# 定义网络结构
class MyNet(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        super(MyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim1), nn.Dropout(0.3)  # 目前最好成绩不加dropout
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim1),
            nn.Linear(hidden_dim1, hidden_dim2), nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim2, out_dim)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# 获取dataloader
def get_dataloader():
    train_data = sio.loadmat('./train(Task_2)/train_data.mat')['train_data']
    train_label = sio.loadmat('./train(Task_2)/train_label.mat')['train_label']
    my_train_data, my_train_label = shuffle_data(train_data, train_label)
    my_train_label = [1 if label == 1 else 0 for label in my_train_label]

    transform = {
        'train': transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()]),
        'valid': transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
    }

    train_dataset = My_dataset(my_train_data, my_train_label, 'train', transform=transform['train'])
    valid_dataset = My_dataset(my_train_data, my_train_label, 'valid', transform=transform['valid'])
    train_only_dataset = My_dataset(my_train_data, my_train_label, 'train_only', transform=transform['valid'])
    train_dataloader = DataLoader(train_dataset, batch_size=512)
    valid_dataloader = DataLoader(valid_dataset, batch_size=512)
    train_only_dataloader = DataLoader(train_only_dataset, batch_size=512)
    return train_dataloader, valid_dataloader, train_only_dataloader


# 使得画图时的标题可以为中文
def set_chinese():
    import matplotlib
    print("[INFO] matplotlib版本为: %s" % matplotlib.__version__)
    matplotlib.rcParams['font.sans-serif'] = ['FangSong']
    matplotlib.rcParams['axes.unicode_minus'] = False


# 可视化dataloader里的图片
def show_pic(dataloader):
    set_chinese()
    examples = enumerate(dataloader)  # 组合成一个索引序列
    batch_idx, (example_data, example_targets) = next(examples)
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.tight_layout()
        img = example_data[i].reshape(19, 19).T
        plt.imshow(img, cmap='gray')
        target = '是人脸' if example_targets[i] == 1 else '不是人脸'
        plt.title(f'target: {target}')
        plt.xticks([])
        plt.yticks([])
    plt.show()


# 既训练也验证
def train_valid(net, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, optim='sgd',
                init=True, scheduler_type='Cosine'):
    # 权重参数初始化
    def init_xavier(m):
        # if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.1)
            nn.init.constant_(m.bias, 0)

    if init:
        net.apply(init_xavier)

    print('training on:', device)
    net.to(device)

    # 优化器选择
    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=1e-5, momentum=0.9)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=1e-4)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=1e-4)

    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)
    elif scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    train_losses = []
    train_acces = []
    eval_acces = []
    best_acc = 0.0
    for epoch in range(num_epoch):

        print("——————第 {} 轮训练开始——————".format(epoch + 1))

        # 训练开始
        net.train()
        train_acc = 0
        for batch in tqdm(train_dataloader):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)
            Loss = loss(output, targets)
            # 优化器优化模型
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / (batch_size)
            train_acc += acc
        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch, Loss.item(), train_acc / len(train_dataloader)))
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(Loss.item())

        # 测试步骤开始
        net.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = net(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss
                acc = num_correct / imgs.shape[0]
                eval_acc += acc

            eval_losses = eval_loss / (len(valid_dataloader))
            eval_acc = eval_acc / (len(valid_dataloader))
            if eval_acc > best_acc:
                best_acc = eval_acc
                best_model_wts = net.state_dict()
            eval_acces.append(eval_acc)
            print("整体验证集上的Loss: {}".format(eval_losses))
            print("整体验证集上的正确率: {}".format(eval_acc))
    net.load_state_dict(best_model_wts)
    torch.save(net, "best_acc.pth")
    return train_losses, train_acces, eval_acces


# 将所有样本都用来训练，效果没有既训练又验证好(可能数据太多导致过拟合)
def train_only(net, loss, train_dataloader, device, batch_size, num_epoch, lr, lr_min, optim='sgd',
               scheduler_type='Cosine', init=True):
    def init_xavier(m):
        # if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    if init:
        net.apply(init_xavier)
    print('training on:', device)
    net.to(device)

    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=1e-4, momentum=0.9)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=1e-4)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=1e-4)

    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

    train_losses = []
    train_acces = []
    best_acc = 0.0
    for epoch in range(num_epoch):

        print("——————第 {} 轮训练开始——————".format(epoch + 1))

        # 训练开始
        net.train()
        train_acc = 0
        for batch in tqdm(train_dataloader):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)
            Loss = loss(output, targets)
            # 优化器优化模型
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / (batch_size)
            train_acc += acc
        #scheduler.step()
        if (train_acc / len(train_dataloader)) > best_acc:
            best_acc = (train_acc / len(train_dataloader))
            torch.save(net, "best_acc.pth")
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch, Loss.item(), train_acc / len(train_dataloader)))
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(Loss.item())
    print('----------训练结束-----------')
    return train_losses, train_acces


# 可视化训练过程的精度
def show_acces(train_losses, train_acces, valid_acces, num_epoch):  # 对准确率和loss画图显得直观
    plt.plot(1 + np.arange(len(train_losses)), train_losses, linewidth=1.5, linestyle='dashed', label='train_losses')
    plt.plot(1 + np.arange(len(train_acces)), train_acces, linewidth=1.5, linestyle='dashed', label='train_acces')
    plt.plot(1 + np.arange(len(valid_acces)), valid_acces, linewidth=1.5, linestyle='dashed', label='valid_acces')
    plt.grid()
    plt.xlabel('epoch')
    plt.xticks(range(1, 1 + num_epoch, 1))
    plt.legend(loc='upper right')
    plt.show()


# 因为原始数据前一半是一类样本，后一半是另一类，所以对划分训练集和验证集不方便，将原始数据打乱再划分保证划分的有效性
def shuffle_data(data, label):
    result = []
    result_data = []
    result_label = []
    for i in range(len(label)):
        temp = np.append(data[i], label[i])
        result.append(temp)
    shuffle(result)

    for i in range(len(result)):
        result_data.append(result[i][:-1])
        result_label.append(result[i][-1])
    return np.array(result_data).astype(np.float32), result_label


if __name__ == '__main__':
    train_dataloader, valid_dataloader, train_only_dataloader = get_dataloader()
    #show_pic(train_dataloader) # 可视化几张数据集图片看看
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Train_only = False
    if Train_only:
        # net = torch.load('best_acc.pth')
        net = MyNet(19 * 19, 500, 500, 2)
        loss = nn.CrossEntropyLoss()
        _, _ = train_only(net, loss, train_only_dataloader, device, batch_size=512, num_epoch=12, lr=0.1, lr_min=1e-4,
                          optim='sgd', init=False)
    else:
        net = MyNet(19 * 19, 400, 400, 2)  # 目前最好 100, 50
        loss = nn.CrossEntropyLoss()
        train_losses, train_acces, eval_acces = train_valid(net, loss, train_dataloader, valid_dataloader, device,
                                                            batch_size=512, num_epoch=13, lr=0.1, lr_min=1e-3,
                                                            optim='sgd', init=False)
        #show_acces(train_losses, train_acces, eval_acces,10) # 展示训练过程

    # 进行任务的测试
    test_data = torch.tensor(sio.loadmat('./test(Task_2)/test_data.mat')['test_data'])
    net = torch.load('best_acc.pth')  # 导入最好一次训练epoch的模型进行预测
    net.eval()
    _, out = net(test_data).max(1)
    # 将结果写入txt文件
    with open('result_mlp.txt', 'w') as f:
        for i in range(len(out)):
            if out[i] == 1:
                f.write(f'{i + 1} 1\n')
            else:
                f.write(f'{i + 1} -1\n')
