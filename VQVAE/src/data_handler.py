# カスタムデータセットクラスの定義
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class DataSet(Dataset):
    def __init__(self, data, transform=False):
        self.X = data[0]
        self.y = data[1]
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = self.X[index].view(28, 28)
        label = self.y[index]
        if self.transform:
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
        return img, label

def get_mnist_dataloaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    x_train = train_dataset.data.reshape(-1, 784).float() / 255
    y_train = F.one_hot(train_dataset.targets, 10).float()
    x_test = test_dataset.data.reshape(-1, 784).float() / 255
    y_test = F.one_hot(test_dataset.targets, 10).float()

    trainset = DataSet([x_train, y_train], transform=transform)
    testset = DataSet([x_test, y_test], transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

    return trainloader, testloader