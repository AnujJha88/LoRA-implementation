import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from LoRA import LoRALayer,set_lora_active,apply_lora_layer
from test_network import SimpleModel,train_model,evaluate_model,count_trainable_params
import time
from LoRA import get_lora_params

# make torch deterministic
_=torch.manual_seed(0)

def main():
        # Generate synthetic dataset
    torch.manual_seed(0)
    train_X = torch.randn(1000, 1, 28, 28)
    train_y = torch.randint(0, 10, (1000,))
    test_X = torch.randn(200, 1, 28, 28)
    test_y = torch.randint(0, 10, (200,))
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=32)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Baseline Model (No LoRA)
    print("\nTraining Baseline Model...")
    baseline_model = SimpleModel().to(device)
    baseline_train_time = train_model(baseline_model, train_loader, device=device)
    baseline_acc = evaluate_model(baseline_model, test_loader, device=device)
    baseline_params = count_trainable_params(baseline_model)

    # LoRA Model
    print("\nTraining LoRA Model...")
    lora_model = SimpleModel().to(device)
    for module in lora_model.modules():
        if isinstance(module, nn.Linear):
            apply_lora_layer(module, rank=3)
    lora_train_time = train_model(lora_model, train_loader, device=device)
    lora_acc = evaluate_model(lora_model, test_loader, device=device)
    lora_params_list = get_lora_params(lora_model)
    lora_params_count = sum(p.numel() for p in lora_params_list)


    # Print results
    print("\n" + "="*40)
    print(f"{'Metric':<20}{'Baseline':<15}{'LoRA':<15}")
    print('='*40)
    print(f"{'Accuracy':<20}{baseline_acc:<15.4f}{lora_acc:<15.4f}")
    print(f"{'Train Time (s)':<20}{baseline_train_time:<15.2f}{lora_train_time:<15.2f}")
    print(f"{'Trainable Params':<20}{baseline_params:<15,}{lora_params_count:<15,}")
    print("\n" + "="*40)

    print("LoRA trainable parameters:")
    for p in get_lora_params(lora_model):
        print(p.shape, p.requires_grad)


if __name__ == '__main__':
    main()