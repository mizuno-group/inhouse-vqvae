# train.py

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model.model import MLP  # model.pyからMLPクラスをインポート
import hydra
from omegaconf import DictConfig
from src.data_handler import get_mnist_dataloaders, DataSet  # DataSet もこちらで定義
from src.model import VQVAE
import wandb

@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wandbの設定
    wandb.init(config=dict(cfg.model),
               entity="benzelongji-the-university-of-tokyo",
               project="2025-8-28-vqvae-mlp")
    
    train_loader, test_loader = get_mnist_dataloaders(cfg.train.batch_size)
    
    # モデル、損失関数、最適化手法の定義
    model = MLP(**cfg.model).to(device)
    # for param in model.vqvae.parameters():
    #     param.requires_grad = False

    # パラメータ数の記録
    total_params = sum(
	param.numel() for param in model.parameters()
    )
    wandb.config.total_parameters = total_params

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    """
    モデル内部に移行したためコメントアウト
    問題なければ消してOK

    # 保存されたモデルのファイルパス
    model_path = "VQVAE_local.pth"
    # VQVAEモデルのインスタンスの作成

    vqvae = VQVAE(128, 32, 2, 256, 64, .25)
    # 保存されたモデルのパラメータをロード
    checkpoint = torch.load(model_path)
    vqvae.load_state_dict(checkpoint['param'])
    # モデルを適切なデバイス（GPUまたはCPU）に移動
    vqvae = vqvae.to(device)
    """    

    # 学習ループ
    for epoch in range(cfg.train.epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # _, _, inputs = vqvae(inputs)
            # 1. viewで元のpermute後の形状に戻す (B, H, W, C)
            # inputs = inputs.view(-1, cfg.model.input_size)#.float()
            # inputs = torch.flatten(inputs, start_dim=1)
            # print(inputs.size())

            # 勾配をゼロにリセット
            optimizer.zero_grad()

            # 順伝播、誤差計算、逆伝播、パラメータ更新
            outputs = model(inputs)
            # print(f"outputs shape: {outputs.shape}") 
            # print(f"labels shape: {labels.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()

        print(f'Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {running_loss/len(train_loader):.4f}')
        wandb.log({"loss": running_loss/len(train_loader)})

        model.eval()  # モデルを評価モードに設定
        correct = 0
        total = 0

        # 推論中は勾配計算を無効にする
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # _, _, inputs = vqvae(inputs)
                # inputs = inputs.view(-1, cfg.model.input_size)#.float()

                # 予測を行う
                outputs = model(inputs)

                # 確率が最も高いクラスのインデックスを取得
                # `torch.max`は (最大値, 最大値のインデックス) のタプルを返す
                predicted = torch.argmax(outputs.data, 1)
                if labels.dim() == 2 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)

                # 全サンプルの総数を更新
                total += labels.size(0)

                # 正しく予測できた数を更新
                correct += (predicted == labels).sum().item()

        # 最終的な精度を計算して出力
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        wandb.log({"accuracy": accuracy})

        # 学習済みモデルの保存
    torch.save(model.state_dict(), 'mlp_mnist.pth')
    print('Finished Training')
    wandb.alert(title="Finished training", text="Finished training")


if __name__ == "__main__":
    main()