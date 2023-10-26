# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)
#
#         # Compute the positional encodings once in log space
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
#
#
# class Transformer(nn.Module):
#     def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
#                  dim_feedforward=2048, dropout=0.1):
#         super(Transformer, self).__init__()
#
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
#                                                         dropout=dropout)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
#
#         self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
#                                                         dropout=dropout)
#         self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
#
#         self.fc_out = nn.Linear(d_model, output_dim)
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         self.positional_encoding = PositionalEncoding(d_model)
#
#     def forward(self, src, tgt):
#         src = self.positional_encoding(src)
#         tgt = self.positional_encoding(tgt)
#
#         memory = self.encoder(src)
#         output = self.decoder(tgt, memory)
#
#         output = self.fc_out(output)
#
#         return output
#
#
# # Example usage
# input_dim = 10
# output_dim = 10
#
# model = Transformer(input_dim, output_dim)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math


# 数据准备
class Seq2SeqDataset(Dataset):
    def __init__(self, seq_length, num_samples):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.data = torch.randn(seq_length, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq = self.data[:, idx]
        target_seq = torch.flip(input_seq, [0])  # 翻转输入序列作为目标序列

        # 裁剪序列，使输入和目标序列具有相同的长度
        min_length = min(input_seq.size(0), target_seq.size(0))
        input_seq = input_seq[:min_length]
        target_seq = target_seq[:min_length]

        return input_seq, target_seq


# 创建数据加载器
seq_length = 10
num_samples = 1000
batch_size = 32
dataset = Seq2SeqDataset(seq_length, num_samples)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 定义 Positional Encoding 类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 定义 Transformer 模型
# 定义 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, output_dim)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, src, tgt):
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.fc_out(output)
        return output

# 训练模型
input_dim = 1
output_dim = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(input_dim, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for input_seq, target_seq in loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        output = model(input_seq.unsqueeze(-1), target_seq.unsqueeze(-1))
        loss = criterion(output, target_seq.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")

# 测试模型
with torch.no_grad():
    input_seq = torch.randn(seq_length, batch_size).to(device)
    target_seq = torch.flip(input_seq, [0])
    output = model(input_seq.unsqueeze(-1), target_seq.unsqueeze(-1))
    print("Input sequence:", input_seq[:, 0])
    print("Target sequence:", target_seq[:, 0])
    print("Predicted sequence:", output[:, 0, 0])