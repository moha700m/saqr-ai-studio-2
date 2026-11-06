"""Generate the SAQR app icon (PNG + multi-size ICO) — original artwork drawn with Pillow."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

root = Path(__file__).resolve().parents[1]
S = 512
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
# rounded-square background with violet -> cyan diagonal gradient
grad = Image.new("RGBA", (S, S))
px = grad.load()
for y in range(S):
    for x in range(S):
        t = (x + y) / (2 * S)
        r = int(139 * (1 - t) + 34 * t); g = int(92 * (1 - t) + 211 * t); b = int(246 * (1 - t) + 238 * t)
        px[x, y] = (r, g, b, 255)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([16, 16, S - 16, S - 16], radius=110, fill=255)
img.paste(grad, (0, 0), mask)
d = ImageDraw.Draw(img)
# stylised falcon head: dark wedge + eye + beak
dark = (14, 14, 22, 255)
d.polygon([(120, 330), (210, 150), (330, 120), (410, 190), (360, 230), (300, 225), (250, 300), (300, 390), (180, 400)], fill=dark)
d.polygon([(360, 230), (410, 190), (420, 250), (385, 265)], fill=(236, 236, 243, 255))
d.ellipse([292, 160, 324, 192], fill=(34, 211, 238, 255))
d.ellipse([302, 170, 314, 182], fill=dark)
out = root / "desktop/saqr_desktop/assets"
img.save(out / "saqr.png")
img.save(out / "saqr.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("icon written")
