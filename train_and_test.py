from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import conf
from get_dataset import MyDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model import MyModel
import torch.optim as optim
import torch


def model_train(load=False, file=""):
    # 读取超参数
    batch_size = conf.batch_size
    device = conf.device
    epochs = conf.epochs

    # 读取训练数据
    training_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    # 构造数据集
    training_data, val_data = torch.utils.data.random_split(training_data, [40000, 20000])  # 划分验证集
    training_set = MyDataset(training_data)  # 实例化训练集对象
    val_set = MyDataset(val_data)  # 实例化验证集对象

    training_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)  # 实例化训练集迭代器
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)  # 实例化验证集迭代器

    # 载入模型
    model = MyModel().to(device)  # 实例化模型对象，并把它放到GPU上
    if load == True:  # 如果有必要，从已经保存好的模型中进行载入
        model.load_state_dict(torch.load(file))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 采用交叉熵损失函数
    optimizer = optim.Adam(model.parameters())  # 采用Adam优化器

    # 定义保存训练指标的字典
    history_train = {'Train Loss': [], 'Train Accuracy': []}  # 保存训练的损失和准确率
    history_val = {'Val Loss': [], 'Val Accuracy': []}  # 保存验证的损失和准确率
    for epoch in range(1, epochs + 1):  # 从1开始计数，方便计数
        # 开始训练
        model.train()  # 转换到训练模式
        total_loss = 0  # 初始化总损失和总准确率
        total_accuracy = 0
        for i, data in enumerate(training_loader):
            x = data[0].to(device)  # 获取图片，并把它放到GPU上
            y = data[1].to(device)  # 获取标签，并把它放到GPU上

            optimizer.zero_grad()  # 梯度清零
            output = model(x)  # 进行预测
            loss = criterion(output, y)  # 计算损失并反向传播
            total_loss += loss  # 损失累加
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            y_hat = torch.argmax(output, dim=1)  # 获取预测结果
            total_accuracy += torch.sum(y_hat == y)  # 累计预测准确的个数

        avg_loss = total_loss / len(training_set)  # 计算平均损失
        accuracy = total_accuracy / len(training_set)  # 计算准确率
        history_train['Train Loss'].append(avg_loss)  # 保存训练损失
        history_train['Train Accuracy'].append(accuracy)  # 保存训练准确率
        print("[%d/%d] |Train Loss: %.8f, Acc: %.4f" % (epoch, epochs, avg_loss, accuracy), end="")  # 提示信息

        # 保存模型
        torch.save(model.state_dict(), "./models/model_epoch{}.pth".format(epoch))

        # 开始验证
        model.eval()  # 调到验证模式
        total_loss = 0
        total_accuracy = 0
        for i, data in enumerate(val_loader):
            x = data[0].to(device)
            y = data[1].to(device)

            with torch.no_grad():
                output = model(x)
                loss = criterion(output, y)

            y_hat = torch.argmax(output, dim=1)
            total_accuracy += torch.sum(y_hat == y)
            total_loss += loss
        avg_loss = total_loss / len(val_set)
        accuracy = total_accuracy / len(val_set)
        history_val['Val Loss'].append(avg_loss)
        history_val['Val Accuracy'].append(accuracy)
        print(" |Val Loss: %.8f, Acc: %.4f" % (avg_loss, accuracy))

    # 开始画图
    Train_Loss = []  # 将放在GPU上的数据放到一个CPU上的list上便于画图
    Train_Acc = []
    Val_Loss = []
    Val_Acc = []
    for i in history_train['Train Loss']:
        Train_Loss.append(i.cpu().detach().numpy())
    for i in history_train['Train Accuracy']:
        Train_Acc.append(i.cpu().detach().numpy())
    for i in history_val['Val Loss']:
        Val_Loss.append(i.cpu().detach().numpy())
    for i in history_val['Val Accuracy']:
        Val_Acc.append(i.cpu().detach().numpy())

    plt.plot(Train_Loss, label="Train Loss")
    plt.plot(Val_Loss, label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    plt.plot(Train_Acc, label="Train Accuracy")
    plt.plot(Val_Acc, label="Val Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.show()


def model_test(file=""):
    batch_size = conf.batch_size
    device = conf.device

    testing_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    testing_set = MyDataset(testing_data)
    testing_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=False)

    model = MyModel().to(device)
    model.load_state_dict(torch.load(file))

    model.eval()
    total_accuracy = 0
    for i, data in enumerate(testing_loader):
        x = data[0].to(device)
        y = data[1].to(device)
        output = model(x)
        y_hat = torch.argmax(output, dim=1)
        total_accuracy += torch.sum(y_hat == y)
    avg_accuracy = total_accuracy / len(testing_set)
    print("Test Acc:%8f" % avg_accuracy)
