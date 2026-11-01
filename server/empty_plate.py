import sys, os, numpy as np, torch, cv2
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
a = np.array(Image.open(src).convert("RGBA"))
rgb, alpha = a[..., :3].copy(), a[..., 3]
mask = (alpha < 128).astype(np.uint8) * 255
if mask.mean() < 1: sys.exit("遮罩是空的：請確認是在『遮罩圖層』塗的")
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49)))
print("遮罩佔畫面 %.1f%%" % (mask.mean() / 2.55))
p = "/app/models/big-lama.pt"
if not os.path.exists(p):
    torch.hub.download_url_to_file("https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt", p)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = torch.jit.load(p, map_location=dev).eval()
h, w = rgb.shape[:2]; H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8
img = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255
msk = torch.from_numpy((mask > 0).astype(np.float32))[None, None]
img = torch.nn.functional.pad(img, (0, W - w, 0, H - h), mode="reflect")
msk = torch.nn.functional.pad(msk, (0, W - w, 0, H - h))
with torch.no_grad():
    r = m(img.to(dev), msk.to(dev))[0, :, :h, :w]
r = (r.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
f = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32)[..., None] / 255
Image.fromarray((r * f + rgb * (1 - f)).astype(np.uint8)).save(out)
print("完成：", out)
