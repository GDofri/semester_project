
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(self, input_dim=32, num_numeric_features=7, embed_dim=128, num_layers=3, num_heads=4, dropout=0.1):
        # super(Transformer, self).__init__()
        super().__init__()

        self.num_numeric_features = num_numeric_features

        self.token_embedding = nn.Linear(input_dim, embed_dim)

        self.numeric_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            for _ in range(num_numeric_features)
        ])

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

    def forward(self, x, numeric_features, attention_mask):
        # x: (batch_size, seq_len, input_dim)
        # numeric_features: (batch_size, num_numeric_features)
        # Mask is False for masked tokens, True for non-masked tokens
        # attention_mask: (batch_size, seq_len)

        batch_size, seq_len, input_dim = x.size()

        numeric_features = numeric_features.unsqueeze(-1) # (batch_size, num_numeric_features, 1)

        numeric_embeddings = []
        for i, numeric_embedding in enumerate(self.numeric_embeddings):
            numeric_feature = numeric_features[:, i, :]
            numeric_feature_embedding = numeric_embedding(numeric_feature)
            numeric_embeddings.append(numeric_feature_embedding)

        # Embedded numeric features # (batch_size, num_numeric_features, embed_dim)
        numeric_features = torch.stack(numeric_embeddings, dim=1)

        # Token embedding
        x = self.token_embedding(x)

        # Concatenate token and numeric embeddings
        x = torch.cat([numeric_features, x], dim=1)

        # Apply positional encoding
        x = self.positional_encoding(x)

        # Add ones to attention mask for numeric tokens
        attention_mask = torch.cat([torch.ones(batch_size, self.num_numeric_features, dtype=torch.bool, device=x.device),
                                    attention_mask], dim=1)

        # Invert attention mask to create key_padding_mask
        key_padding_mask = ~attention_mask

        # True for masked tokens, False for non-masked tokens
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # Pooling and regression head
        x = x.mean(dim=1) # (batch_size, embed_dim)
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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
