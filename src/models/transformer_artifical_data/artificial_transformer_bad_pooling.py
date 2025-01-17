
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.base_model import BaseModel


class TransformerBadPool(BaseModel):
    def __init__(self, input_dim=32, embed_dim=128, num_layers=3, num_heads=4, dropout=0.1):
        # super(Transformer, self).__init__()
        super(TransformerBadPool, self).__init__(
            name="TransformerNoNumericBadPooling",
            description="Basic transformer w. masked input. No masked pooling",
            input_requires_mask=True,
            input_requires_numerics=False,
            supports_variable_sequence_length=True
        )

        self.token_embedding = nn.Linear(input_dim, embed_dim)

        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=embed_dim*4, dropout=dropout,
                                                   activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1)
        )

    def forward(self, x, attention_mask):
        # x: (batch_size, seq_len, input_dim)
        # numeric_features: (batch_size, num_numeric_features)
        # Mask is False for masked tokens, True for non-masked tokens
        # attention_mask: (batch_size, seq_len)

        batch_size, seq_len, input_dim = x.size()

        # Token embedding
        x = self.token_embedding(x)

        # Apply positional encoding
        x = self.positional_encoding(x)

        # Invert attention mask to create key_padding_mask
        # True for masked tokens, False for non-masked tokens
        key_padding_mask = ~attention_mask

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # # Apply masking before pooling
        # mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        # x = x * mask  # Zero out the masked positions
        #
        # # Compute the sum over the sequence dimension
        # x_sum = x.sum(dim=1)  # (batch_size, embed_dim)
        #
        # # Compute the number of valid tokens per example
        # valid_token_count = mask.sum(dim=1)  # (batch_size, 1)
        #
        # # Avoid division by zero
        # valid_token_count = valid_token_count.clamp(min=1)
        #
        # # Compute the mean over valid tokens
        # x_mean = x_sum / valid_token_count  # (batch_size, embed_dim)
        #
        # # Pass through the regression head
        # output = self.regression_head(x_mean)  # (batch_size, 1)
        #
        # return output

        # Pooling and regression head
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        output = self.regression_head(x)  # (batch_size, 1)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

