import torch
from torchvision import datasets, transforms


# 数据集全局加载工具类
class DataSetUtil:
    # 需要将数据变为tensor format
    dataTans = transforms.Compose([
        transforms.ToTensor()
    ])

    def initDataSet(self):
        # 自动下载训练数据集
        train_dataset = datasets.MNIST(root='./data', train=True, transform=self.dataTans, download=True)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
        # 自动下载测试数据集
        test_dataset = datasets.MNIST(root='./data', train=False, transform=self.dataTans, download=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
        return train_dataloader, test_dataloader, train_dataset, test_dataset
