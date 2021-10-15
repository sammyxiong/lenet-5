import torch
from network import LeNet
from torch import nn
from datasetutil import DataSetUtil
from torch.optim import lr_scheduler
import os

# 这个文件用作训练

# 获取数据集
train_dataloader, test_dataloader, _, _ = DataSetUtil().initDataSet()

# 如果有GPU，可以转GPU运行
device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"
model = LeNet().to(device)

# 定义损失函数-交叉熵
loss_fn = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每10轮变为0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定义forward propagation函数
def forward(pair):
    x, y = pair[0].to(device), pair[1].to(device)
    output = model(x)
    cur_loss = loss_fn(output, y)
    _, pred = torch.max(output, axis=1)
    cur_acc = torch.sum(y == pred) / output.shape[0]
    return cur_loss, cur_acc


# 定义训练函数
def train(dataloader, optimize):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        cur = forward((x, y))
        optimize.zero_grad()
        cur[0].backward()
        optimize.step()
        loss += cur[0].item()
        current += cur[1].item()
        n = n + 1
    print("train loss:" + str(loss / n) + "，train acc:" + str(current / n))


def verify(dataloader, imode):
    imode.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            cur = forward((x, y))
            loss += cur[0].item()
            current += cur[1].item()
            n = n + 1
        print("verify loss" + str(loss / n) + ",verify acc:" + str(current / n))
        return current / n


# 训练多少次
times = 30
minacc = 0
modelfolder = "./data/model"
for t in range(times):
    print(f'Training No.{t + 1} start\n')
    train(train_dataloader, optimizer)
    a = verify(test_dataloader, model)
    # 保存最好的模型
    if a > minacc:
        if not os.path.exists(modelfolder):
            os.mkdir(modelfolder)
        minacc = a
        torch.save(model.state_dict(), modelfolder + "/model.pth")
print("All training tasks have finished!")
