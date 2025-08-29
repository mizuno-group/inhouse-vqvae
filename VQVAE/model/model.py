# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import VQVAE

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # MLPのパラメータ
        self.number_of_layers = kwargs['number_of_layers']
        self.input_size = kwargs['input_size']
        self.top_hidden_size = kwargs['top_hidden_size']
        self.bottom_hidden_size = kwargs['bottom_hidden_size']
        self.output_size = kwargs['output_size']
        self.emb_dim = kwargs['embedding_dim']

        # VQVAEのパラメータ
        self.vq_h_dim = kwargs['vqvae_h_dim']
        self.vq_res_h_dim = kwargs['vqvae_res_h_dim']
        self.vq_n_res = kwargs['vqvae_n_res_layers']
        self.vq_n_emb = kwargs['vqvae_n_embeddings']
        self.vq_emb_dim = kwargs['vqvae_embedding_dim']
        self.vq_beta = kwargs['vqvae_beta']
        # self.vqvae_path = kwargs['vqvae_path']

        # VQVAEの用意
        self.vqvae = VQVAE(self.vq_h_dim,
                           self.vq_res_h_dim,
                           self.vq_n_res,
                           self.vq_n_emb
                           self.vq_emb_dim,
                           self.vq_beta,)
        
        checkpoint = torch.load('VQVAE_local.pth', weights_only=True)
        self.vqvae.load_state_dict(checkpoint['param'])
        
        # 埋め込み層の用意
        self.embedding = nn.Embedding(self.vq_n_emb, self.emb_dim)

        # その他MLPの用意
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(self.input_size*self.emb_dim, self.top_hidden_size))
        self.layers.append(nn.ReLU())
        for i in range(int((self.number_of_layers-2)/2)):
            self.layers.append(nn.Linear(self.top_hidden_size, self.top_hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.top_hidden_size, self.bottom_hidden_size))
        self.layers.append(nn.ReLU())
        for i in range(int((self.number_of_layers-3)/2)):
            self.layers.append(nn.Linear(self.bottom_hidden_size, self.bottom_hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.bottom_hidden_size, self.output_size))
    

    def forward(self, x):
        # VQVAEを通し潜在表現を取得
        _, _, x = self.vqvae(x)
        x = x.view(-1, self.input_size)

        # 埋め込みを行う
        x = self.embedding(x)

        # 次元のベクトルに変換
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = self.layers(x)
        # ソフトマックスは損失関数内で計算されるため、ここでは適用しない
        return x