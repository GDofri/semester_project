import torch
import torch.nn as nn
import math
from src.models.base_model import BaseModel

class NumericOnlyRegressor(BaseModel):
    def __init__(self, num_numeric_features=7, hidden_dim=128):
        super(NumericOnlyRegressor, self).__init__(
            name="NumericOnlyRegressor",
            description="A simple model that does regression on the 7 numeric features only.",
            input_requires_mask=False,
            input_requires_numerics=True,
            supports_variable_sequence_length=False
        )

        # Example: a small feedforward network for the regression
        self.regression_head = nn.Sequential(
            nn.Linear(num_numeric_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x, numeric_features):
        # numeric_features: shape (batch_size, 7)
        output = self.regression_head(numeric_features)
        return output