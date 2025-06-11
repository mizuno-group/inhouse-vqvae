import torch
import torch.nn.functional as F
from torch import optim, nn
import matplotlib.pyplot as plt

from src.model import VQVAE
from src.trainer import Trainer
from src.data_handler import get_mnist_dataloaders, DataSet  # DataSet もこちらで定義
from src.utils import plot_loss
import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="benzelongji-the-university-of-tokyo",
    # Set the wandb project where this run will be logged.
    project="250610_VQVAE_Test",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 3e-4,
        "architecture": "VQVAE",
        "dataset": "MNIST",
        "epochs": 6,
        "BSZ": 256,
    },
)


# 設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256
max_epoch = 5
learning_rate = 3e-4

# データローダーの取得
trainloader, testloader = get_mnist_dataloaders(batch_size)

# モデルの初期化
model = VQVAE(128, 32, 2, 512, 64, .25).to(device)
opt = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9))

# Trainer の初期化
trainer = Trainer(model, opt, device, trainloader, testloader, max_epoch, run)

# 学習の実行
train_loss_log, test_loss_log = trainer.train()

# 結果のプロット
plot_loss(train_loss_log, test_loss_log)

# 最終エポックでモデルを保存 (Trainer クラス内で行うことも可能です)
torch.save({'param': model.to('cpu').state_dict(),
            'opt': opt.state_dict(),
            'epoch': trainer.epoch},
           'VQVAE_local.pth')