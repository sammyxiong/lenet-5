import torch
from network import LeNet
from torch.autograd import Variable
from datasetutil import DataSetUtil

# 这个文件用做测试

# 获取数据集
_, _, train_dataset, test_dataset = DataSetUtil().initDataSet()

# 结果数组
results = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 如果有GPU，可以转GPU运行
device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"
model = LeNet().to(device)


# 训练好的文件路径
modelLoc = "./data/model/model.pth"
# 加载训练好的模型文件
model.load_state_dict(torch.load(modelLoc))
model.eval()

# 拿测试数据来测试一下
testCnt = 5
# 识别正确的次数
validCnt = 0
for i in range(testCnt):
    x, y = test_dataset[i][0], test_dataset[i][1]
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x)
        p, a = results[torch.argmax(pred[0])], results[y]
        if p == a:
            validCnt += 1
        print(f'real number:"{a}",machine predicted:"{p}"')
print(f'The detection success rate is:{validCnt * 1.0 / testCnt * 100}%')
