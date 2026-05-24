"""Spandrel-based upscaler loader.

One model loaded at boot. Tiled inference so we can upscale large inputs
without blowing VRAM. No CPU offload — on GB10 unified memory it would just
copy bytes around for no gain.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

log = logging.getLogger("inemaupsk.loader")


class UpscaleLoader:
    @classmethod
    def load(cls, weight_path: str) -> ImageModelDescriptor:
        p = Path(weight_path)
        if not p.exists():
            raise FileNotFoundError(f"weights not found: {weight_path}")
        model = ModelLoader().load_from_file(str(p))
        if not isinstance(model, ImageModelDescriptor):
            raise TypeError(f"unsupported model type for {weight_path}: {type(model)}")
        model.cuda().eval()
        log.info(
            "loaded %s scale=%dx arch=%s",
            p.name, model.scale, type(model.model).__name__,
        )
        return model

    @classmethod
    @torch.inference_mode()
    def upscale(
        cls,
        model: ImageModelDescriptor,
        img: Image.Image,
        tile: int = 512,
        tile_pad: int = 16,
    ) -> Image.Image:
        """Tiled upscale. tile=0 disables tiling (single pass)."""
        img = img.convert("RGB")
        t = _to_tensor(img).cuda()
        if tile <= 0 or max(img.size) <= tile:
            out = model(t)
        else:
            out = _tiled_forward(model, t, tile=tile, pad=tile_pad)
        return _to_pil(out)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np
    arr = np.asarray(img, dtype="float32") / 255.0  # H,W,C
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    return t


def _to_pil(t: torch.Tensor) -> Image.Image:
    import numpy as np
    t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    arr = (t * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _tiled_forward(
    model: ImageModelDescriptor,
    t: torch.Tensor,
    tile: int,
    pad: int,
) -> torch.Tensor:
    """Process the input in overlapping tiles, stitch with hard cuts at tile
    interiors (pad rim is what absorbs seam artifacts)."""
    scale = model.scale
    _, c, h, w = t.shape
    out = torch.zeros((1, c, h * scale, w * scale), dtype=t.dtype, device=t.device)

    n_y = math.ceil(h / tile)
    n_x = math.ceil(w / tile)
    for iy in range(n_y):
        for ix in range(n_x):
            y0 = iy * tile
            x0 = ix * tile
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)

            py0 = max(y0 - pad, 0)
            px0 = max(x0 - pad, 0)
            py1 = min(y1 + pad, h)
            px1 = min(x1 + pad, w)

            patch = t[:, :, py0:py1, px0:px1]
            sr = model(patch)

            # crop the padded rim from the SR output
            top = (y0 - py0) * scale
            left = (x0 - px0) * scale
            bot = top + (y1 - y0) * scale
            right = left + (x1 - x0) * scale
            sr_core = sr[:, :, top:bot, left:right]

            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = sr_core
    return out
