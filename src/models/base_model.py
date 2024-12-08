import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, name: str, description: str, input_requires_mask: bool, input_requires_numerics: bool, supports_variable_sequence_length: bool):
        super().__init__()
        self.name = name
        self.description = description
        self.input_requires_mask = input_requires_mask
        self.input_requires_numerics = input_requires_numerics
        self.supports_variable_sequence_length = supports_variable_sequence_length

    def forward(self, *args, **kwargs):
        """
        This should be overridden in derived classes. The method can
        enforce specific input conditions based on `input_requires_mask`.
        """
        raise NotImplementedError("Forward method must be implemented in the derived class.")

    def get_metadata(self):
        """
        Returns metadata about the model for easier querying.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_requires_mask": self.input_requires_mask,
            "input_requires_numerics": self.input_requires_numerics,
            "supports_variable_sequence_length": self.supports_variable_sequence_length
        }