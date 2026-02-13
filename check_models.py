import torch
import os
from src.model import get_model
from src.ensemble import create_ensemble

def check_if_trained(model_type, model_path):
    if not os.path.exists(model_path):
        print(f"{model_type}: ❌ File not found")
        return

    trained_state = torch.load(model_path, map_location="cpu")

    # Create fresh model
    if "ensemble" in model_type:
        ensemble_type = model_type.replace("ensemble_", "")
        fresh_model = create_ensemble(ensemble_type)
    else:
        fresh_model = get_model(model_type)

    fresh_state = fresh_model.state_dict()

    total_diff = 0.0
    param_count = 0

    for key in fresh_state:
        if key in trained_state:
            diff = torch.sum(torch.abs(fresh_state[key] - trained_state[key])).item()
            total_diff += diff
            param_count += fresh_state[key].numel()

    avg_diff = total_diff / param_count

    if avg_diff < 1e-6:
        print(f"{model_type}: ❌ NOT TRAINED (weights nearly identical)")
    else:
        print(f"{model_type}: ✅ TRAINED")
        print(f"   Avg Weight Difference: {avg_diff:.6f}")
