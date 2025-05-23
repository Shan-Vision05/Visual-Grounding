import math
from torch.nn import Module
import torch
import torch.nn as nn

class PositionalEncoding(Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model



    def forward(self, x):
        '''
        Assuming the input shape includes batch_size dimension, which it should.
        '''
        # print('In Pos Encoding')
        # print(self.posEnc[None,:, :].shape)
        # print(x.shape)
        position_indx = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        posEnc = torch.zeros(self.max_seq_len, self.d_model)

        scale_term  = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        posEnc[:, 0::2] = torch.sin(position_indx * scale_term)
        posEnc[:, 1::2] = torch.cos(position_indx * scale_term)

        embedded = x + posEnc.to(x.device)
        # print(embedded.shape)
        return embedded

class SubjectTextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead = 2, num_layers = 1, dim_ff= 512, max_len=15):
        super().__init__()

        self.TokenEmbed = nn.Embedding(vocab_size, d_model)
        self.PositionEmbedding = PositionalEncoding(d_model, max_len)

        self.EncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            activation='gelu'
        )

        self.TransformerEncoder = nn.TransformerEncoder(self.EncoderLayer, num_layers=num_layers)

        self.ClassificationHead = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_len*d_model, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(768, 512)
        )

    def forward(self, input_ids, attention_mask=None):

        # self.TokenEmbed = self.TokenEmbed.to(input_ids.device)
        # self.TransformerEncoder = self.TransformerEncoder.to(input_ids.device)
        # self.ClassificationHead =  self.ClassificationHead.to(input_ids.device)
        x = self.TokenEmbed(input_ids)
        x = self.PositionEmbedding(x)

        x = x.transpose(0,1)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        x = self.TransformerEncoder(x, src_key_padding_mask=src_key_padding_mask)

        x = self.ClassificationHead(x.transpose(0, 1))
        return x
    
class LocationTextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead = 2, num_layers = 1, dim_ff= 512, max_len=15):
        super().__init__()

        self.TokenEmbed = nn.Embedding(vocab_size, d_model)
        self.PositionEmbedding = PositionalEncoding(d_model, max_len)

        self.EncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            activation='gelu'
        )

        self.TransformerEncoder = nn.TransformerEncoder(self.EncoderLayer, num_layers=num_layers)

        self.ClassificationHead = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_len*d_model, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(768, 512)
        )

    def forward(self, input_ids, attention_mask=None):

        # self.TokenEmbed = self.TokenEmbed.to(input_ids.device)
        # self.TransformerEncoder = self.TransformerEncoder.to(input_ids.device)
        # self.ClassificationHead =  self.ClassificationHead.to(input_ids.device)
        x = self.TokenEmbed(input_ids)
        x = self.PositionEmbedding(x)

        x = x.transpose(0,1)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        x = self.TransformerEncoder(x, src_key_padding_mask=src_key_padding_mask)

        x = self.ClassificationHead(x.transpose(0, 1))
        return x

