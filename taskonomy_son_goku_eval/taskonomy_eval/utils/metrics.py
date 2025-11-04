
from typing import Dict
import torch
import torch.nn.functional as F

def depth_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None) -> Dict[str, float]:
    """
    pred, target: (B,1,H,W) in meters
    """
    if mask is None:
        mask = torch.isfinite(target) & (target > 0)
    mask = mask & torch.isfinite(pred)
    m = mask.float()
    diff = (pred - target).abs() * m
    n = m.sum().clamp_min(1.0)
    rmse = torch.sqrt(((pred - target)**2 * m).sum() / n).item()
    mae = (diff.sum() / n).item()
    rel = ((diff / target.clamp_min(1e-6)) * m).sum() / n
    return {"rmse": float(rmse), "mae": float(mae), "absrel": float(rel.item())}

def normal_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str,float]:
    """
    pred, target: (B,3,H,W) in [-1,1]; not necessarily unit
    """
    eps = 1e-6
    p = F.normalize(pred, dim=1, eps=eps)
    t = F.normalize(target, dim=1, eps=eps)
    cos = (p * t).sum(dim=1).clamp(-1,1)  # (B,H,W)
    ang = torch.rad2deg(torch.acos(cos))
    valid = torch.isfinite(ang)
    m = valid.float()
    n = m.sum().clamp_min(1.0)
    mean = (ang * m).sum() / n
    med = ang.median()
    r11 = ((ang < 11.25) * m).sum() / n
    r22 = ((ang < 22.5) * m).sum() / n
    r30 = ((ang < 30.0) * m).sum() / n
    return {"mean_deg": float(mean.item()), "median_deg": float(med.item()), "11.25deg": float(r11.item()), "22.5deg": float(r22.item()), "30deg": float(r30.item())}

def bce_f1(pred_logits: torch.Tensor, target_mask: torch.Tensor, thresh: float=0.5) -> Dict[str,float]:
    """
    pred_logits: (B,1,H,W) raw logits
    target_mask: (B,1,H,W) in {0,1}
    """
    prob = torch.sigmoid(pred_logits)
    pred = (prob >= thresh).float()
    tp = (pred * target_mask).sum()
    fp = (pred * (1 - target_mask)).sum()
    fn = ((1 - pred) * target_mask).sum()
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    bce = F.binary_cross_entropy(prob.clamp(1e-6, 1-1e-6), target_mask.float())
    return {"f1": float(f1.item()), "precision": float(prec.item()), "recall": float(rec.item()), "bce": float(bce.item())}

def miou(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str,float]:
    """
    pred_logits: (B,C,H,W), target: (B,H,W) with class ids [0..C-1]
    """
    pred = pred_logits.argmax(dim=1)  # (B,H,W)
    ious = []
    eps=1e-6
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union < 1:
            continue
        ious.append((inter + eps) / (union + eps))
    if len(ious)==0:
        return {"miou": 0.0}
    miou = torch.stack(ious).mean().item()
    return {"miou": float(miou)}
