import torch
import torch.nn as nn

from typing import List, Tuple
from torch.utils.data import DataLoader


def train_one_epoch(
    obj_model: nn.Module,
    obj_loader: DataLoader,
    obj_optimizer: torch.optim.Optimizer,
    obj_loss_fn: nn.Module,
    str_device: str
) -> float:
    obj_model.train()
    float_total_loss = 0.0

    for ts_input_ids, ts_labels in obj_loader:
        ts_input_ids = ts_input_ids.to(str_device)
        ts_labels = ts_labels.to(str_device)

        obj_optimizer.zero_grad()
        ts_logits = obj_model(ts_input_ids)
        ts_loss = obj_loss_fn(ts_logits, ts_labels)
        ts_loss.backward()
        obj_optimizer.step()

        float_total_loss += ts_loss.item()

    return float_total_loss / max(len(obj_loader), 1)


def evaluate(
    obj_model: nn.Module,
    obj_loader: DataLoader,
    str_device: str
) -> Tuple[List[int], List[int]]:
    obj_model.eval()

    list_true = []
    list_pred = []

    with torch.no_grad():
        for ts_input_ids, ts_labels in obj_loader:
            ts_input_ids = ts_input_ids.to(str_device)
            ts_logits = obj_model(ts_input_ids)
            ts_pred = torch.argmax(ts_logits, dim=1).cpu().tolist()

            list_pred.extend(ts_pred)
            list_true.extend(ts_labels.tolist())

    return list_true, list_pred
