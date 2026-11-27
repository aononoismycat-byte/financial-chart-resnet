"""Grad-CAM attribution and chart-alignment metrics for ResNet-18 (P1 Stage 6)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation


def gradcam(model, images: torch.Tensor, target_class: int = 1) -> np.ndarray:
    """Compute per-image Grad-CAM over model.backbone.layer4 in [0, 1], shape (B, 224, 224).

    Clones images with requires_grad=True so gradients flow even when all backbone
    parameters are frozen.
    """
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def _forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = model.backbone.layer4.register_forward_hook(_forward_hook)
    try:
        x = images.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        score = logits[:, target_class].sum()
        score.backward()
        acts = activations["value"].detach()
        grads = gradients["value"].detach()
    finally:
        handle.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False).squeeze(1)
    cam_np = cam.cpu().numpy().astype(np.float32)
    max_vals = cam_np.max(axis=(1, 2), keepdims=True)
    safe_max = np.where(max_vals > 0.0, max_vals, 1.0)
    return np.where(max_vals > 0.0, cam_np / safe_max, 0.0)


def right_quarter_share(cam: np.ndarray) -> float | np.ndarray:
    """Fraction of CAM mass falling in columns x >= 168 (most recent ~5 sessions)."""
    arr = np.asarray(cam, dtype=float)
    if arr.ndim == 2:
        total = float(arr.sum())
        return float(arr[:, 168:].sum() / total) if total > 0.0 else 0.0
    if arr.ndim == 3:
        totals = arr.sum(axis=(1, 2))
        right = arr[:, :, 168:].sum(axis=(1, 2))
        return np.where(totals > 0.0, right / np.where(totals > 0.0, totals, 1.0), 0.0)
    raise ValueError(f"Expected 2D or 3D CAM array, got shape {arr.shape}")


def line_lift(cam: np.ndarray, image) -> float:
    """Ratio of CAM mass share on dilated chart line pixels to dilated pixel area share."""
    cam_arr = np.asarray(cam, dtype=float)
    if cam_arr.ndim != 2:
        raise ValueError(f"Expected 2D CAM for line_lift, got shape {cam_arr.shape}")
    img_arr = np.asarray(image)
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {img_arr.shape}")
    if np.issubdtype(img_arr.dtype, np.floating) and img_arr.max() <= 1.0:
        img_uint8 = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img_arr.astype(np.uint8)
    line_pixels = np.any(img_uint8 < 128, axis=-1)
    dilated = binary_dilation(line_pixels, iterations=5)
    total_cam = float(cam_arr.sum())
    area_share = float(dilated.mean())
    if total_cam <= 0.0 or area_share <= 0.0:
        return 0.0
    cam_share = float(cam_arr[dilated].sum()) / total_cam
    return float(cam_share / area_share)
