"""One-off benchmark: upscale one image with N models, save each output and a
2x2 side-by-side comparison crop. Run via `docker exec inemaupsk python /app/bench.py`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/app")
from loaders.upscale import UpscaleLoader  # noqa: E402

MODELS_DIR = Path("/app/models/upscale")
OUT_DIR = Path("/app/outputs")
MODELS = [
    "4x-UltraSharp.pth",
    "RealESRGAN_x4plus.pth",
    "4x_NMKD-Siax_200k.pth",
    "4x_foolhardy_Remacri.pth",
]


def main(src_path: str) -> None:
    src = Image.open(src_path).convert("RGB")
    print(f"input: {src.size} from {src_path}")
    stem = Path(src_path).stem

    outputs: list[tuple[str, Image.Image, float]] = []
    for name in MODELS:
        path = MODELS_DIR / name
        if not path.exists():
            print(f"  skip {name} (not found)")
            continue
        t0 = time.perf_counter()
        model = UpscaleLoader.load(str(path))
        load_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = UpscaleLoader.upscale(model, src, tile=512, tile_pad=16)
        infer_t = time.perf_counter() - t0

        out_path = OUT_DIR / f"{stem}__{Path(name).stem}.png"
        out.save(out_path)
        print(f"  {name}: load={load_t:.1f}s infer={infer_t:.2f}s -> {out_path.name}")
        outputs.append((Path(name).stem, out, infer_t))

        del model
        import torch
        torch.cuda.empty_cache()

    # Build a 2x2 comparison of the SAME 400x400 region at output scale.
    if len(outputs) >= 2:
        # pick a centered crop window in OUTPUT coords
        w_out, h_out = outputs[0][1].size
        cw, ch = min(500, w_out), min(500, h_out)
        x0 = (w_out - cw) // 2
        y0 = (h_out - ch) // 2

        cols = 2
        rows = (len(outputs) + 1) // 2
        label_h = 28
        grid = Image.new("RGB", (cw * cols, (ch + label_h) * rows), "black")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(grid)
        for i, (name, im, infer_t) in enumerate(outputs):
            r, c = divmod(i, cols)
            crop = im.crop((x0, y0, x0 + cw, y0 + ch))
            grid.paste(crop, (c * cw, r * (ch + label_h) + label_h))
            draw.rectangle(
                (c * cw, r * (ch + label_h), c * cw + cw, r * (ch + label_h) + label_h),
                fill="black",
            )
            draw.text(
                (c * cw + 8, r * (ch + label_h) + 4),
                f"{name}  ({infer_t:.2f}s)",
                fill="white", font=font,
            )
        grid_path = OUT_DIR / f"{stem}__compare.png"
        grid.save(grid_path)
        print(f"comparison grid -> {grid_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/outputs/input.png")
