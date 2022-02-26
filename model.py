import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3)),  # 输入为(batch_size,1,28,28),输出为(batch_size,8,26,26)
            nn.Sigmoid(),
            nn.Flatten(),  # 将卷积的结果展平，其长度为26*26*64
            nn.Linear(26 * 26 * 64, 10)  # MNIST的类别有10个，因此输出维度为10
        )

    def forward(self, x):
        return self.net(x)
