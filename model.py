import torch
import torch.nn as nn
import numpy as np


# Transformer模型相关实现
class PositionalEncoding(nn.Module):
    """位置编码模块，为序列添加位置信息"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)


class DecoderOnlyTransformer(nn.Module):
    """仅包含解码器的Transformer模型，类似GPT架构"""

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使用batch_first=True以便与DataLoader兼容
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出层，将模型输出映射到词汇表大小
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        """初始化模型权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, tgt, tgt_mask=None):
        """
        前向传播

        参数:
            tgt: 目标序列 (batch_size, tgt_len)
            tgt_mask: 目标序列掩码，防止模型看到未来的token
        """
        # 嵌入和位置编码
        tgt = self.embedding(tgt) * np.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # Transformer解码器（使用自身作为memory）
        output = self.transformer_decoder(tgt, tgt, tgt_mask=tgt_mask)

        # 输出层
        output = self.fc_out(output)

        return output

    def generate(self, prompt, max_length, pad_token, eos_token, temperature=1.0):
        """
        生成序列

        参数:
            prompt: 提示序列 (batch_size, prompt_len)
            max_length: 最大生成长度
            pad_token: padding标记
            eos_token: 结束标记
            temperature: 温度参数，控制生成的随机性

        返回:
            生成的序列
        """
        self.eval()

        # 复制提示作为初始输入
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # 如果所有序列都已生成结束标记，则停止
                if (generated == eos_token).all(dim=1).all():
                    break

                # 创建目标掩码
                tgt_len = generated.size(1)
                tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(generated.device)

                # 前向传播
                output = self.forward(generated, tgt_mask=tgt_mask)

                # 获取最后一个时间步的输出
                next_token_logits = output[:, -1, :] / temperature

                # 计算概率
                next_token_probs = torch.softmax(next_token_logits, dim=-1)

                # 采样下一个token
                next_token = torch.multinomial(next_token_probs, num_samples=1)

                # 将生成的token添加到序列中
                generated = torch.cat([generated, next_token], dim=1)

                # 对于已经生成结束标记的序列，用padding填充
                eos_mask = (generated == eos_token).any(dim=1)
                if eos_mask.any():
                    next_token[eos_mask] = pad_token

        self.train()
        return generated

    def _generate_square_subsequent_mask(self, sz):
        """生成后续掩码，防止模型看到未来的token"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask