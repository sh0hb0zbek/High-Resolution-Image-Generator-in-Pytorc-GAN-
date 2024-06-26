import torch
from ensure_dir import ensure_dir_for_file


def save_model(model, path, epoch_i):
    model_path = f"{path}_{epoch_i:03d}.pt"
    ensure_dir_for_file(model_path)
    torch.save(model, model_path)