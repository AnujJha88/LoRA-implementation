import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils import parameters_to_vector

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=5, alpha=1.1):
        super().__init__()
        # Create parameters on CPU first
        self.lora_A = nn.Parameter(torch.empty(in_dim, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_dim))
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.active = True
        
        # Initialize parameters
        nn.init.kaiming_normal_(self.lora_A, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.lora_B)  # Initialize B to zeros (common practice)

    def forward(self, original_weight):
        if not self.active:
            return original_weight
        # Compute LoRA update: W = W_orig + (A @ B).T * scale
        delta = self.lora_A @ self.lora_B
        return original_weight + delta.T * self.scale


def apply_lora_layer(layer, rank=2, alpha=1):
    if not isinstance(layer, nn.Linear):
        raise ValueError("Please only use linear layers")

    # Create LoRA layer and move to same device as the original layer
    lora_layer = LoRALayer(layer.in_features, layer.out_features, rank, alpha)
    lora_layer = lora_layer.to(layer.weight.device)
    
    # Apply the parametrization to the weight
    parametrize.register_parametrization(
        layer,
        "weight",
        lora_layer
    )

    # Freeze original weights
    for name, param in layer.named_parameters():
        if 'lora' not in name:
            param.detach_()
            param.requires_grad = False


def set_lora_active(model, active=True):
    for module in model.modules():
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            for param in module.parametrizations.weight:
                if hasattr(param, 'active'):
                    param.active = active


def get_lora_params(model):
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_params.append(param)
    return lora_params