"""自訂學伴：使用者上傳照片（本人同意），做成會說話的數位人，只能當「學伴」。

流程（背景一次做一個）：
  1. 照片 → 1280×720 畫布（rembg 去背後放到房間背景；沒裝 rembg 就照片置中、模糊背景補滿）
  2. 動起來：DGX 主機上的小幫手（tools/lp_worker.sh）用 xiaozhi-liveportrait 的 idle_gen.py
     做 8 秒待機影片（眨眼、頭微微擺動、眼神飄一下）；小幫手沒開或失敗就做「只有嘴巴動」的靜態版
  3. 交給 wav2lip 的 generate_avatar 做成人物資料（data/avatars/custom_<id>）
  4. （選用，CUSTOM_WATERMARK=1）人物畫面角落印上「AI 生成」；預設由網頁標題的「AI 生成」標籤標示

API（都要先過通關碼）：
  POST /custom/create   multipart：photo、name、consent_self、consent_adult、consent_public → {id, key}
  GET  /custom/status?id=
  GET  /custom/thumb?id=
  POST /custom/delete   {id, key}

隱私：id 是亂數，伺服器不提供清單；刪除要建立時拿到的 key（只存在建立者的瀏覽器）。

環境變數：
  CUSTOM_IMG_SIZE   wav2lip 臉部大小，要跟小晴一樣（預設 256）
  LP_TIMEOUT        等小幫手做待機影片最多幾秒（預設 900）
  LP_CMD            （選用）在這個容器裡直接執行的 LivePortrait 指令，可用 {source} {driving} {outdir}；
                    沒設定就交給主機上的小幫手
  LP_DRIVING        LP_CMD 用的動作範本影片（預設 data/custom/driving.mp4）
  CUSTOM_SEG        去背方式：auto（預設，用 rembg；沒裝就退回模糊背景）、key、off
  CUSTOM_BG         背景圖（預設 data/custom/bg.jpg；沒有就用素色牆面）
  CUSTOM_BG_COLOR   素色牆面的顏色（預設 #E8E3DE 淺暖灰）
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import logging
import os
import pickle
import queue
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time

from aiohttp import web

from utils.logger import logger

CUSTOM_DIR = os.path.abspath("data/custom")
AVATAR_DIR = os.path.abspath("data/avatars")
ID_RE = re.compile(r"^custom_[0-9a-f]{16}$")
W, H = 1280, 720
IMG_SIZE = int(os.getenv("CUSTOM_IMG_SIZE", "256"))
LP_CMD = os.getenv("LP_CMD", "").strip()
LP_DRIVING = os.getenv("LP_DRIVING", os.path.join(CUSTOM_DIR, "driving.mp4"))
LP_TIMEOUT = int(os.getenv("LP_TIMEOUT", "900"))
LP_HEARTBEAT = os.path.join(CUSTOM_DIR, ".lp_worker")     # 小幫手每幾秒更新一次這個檔
MAX_PHOTO = 12 * 1024 * 1024
MAX_NAME = 12
FPS = 25
CUSTOM_SEG = os.getenv("CUSTOM_SEG", "auto").strip().lower()   # auto(=rembg) | rembg | key（單色背景，實驗用）| off
CUSTOM_WATERMARK = os.getenv("CUSTOM_WATERMARK", "0") == "1"   # 畫面上印字；網頁標題已經有「AI 生成」標籤，預設不印
CUSTOM_SR = os.getenv("CUSTOM_SR", "1") == "1"        # 小照片先用 Real-ESRGAN 放大變清楚
CUSTOM_SR_URL = os.getenv("CUSTOM_SR_URL",
                          "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth")
CUSTOM_SR_MODEL = os.path.join(CUSTOM_DIR, ".models", "realesr-general-x4v3.pth")
CUSTOM_BG = os.getenv("CUSTOM_BG", os.path.join(CUSTOM_DIR, "bg.jpg"))
os.environ.setdefault("U2NET_HOME", os.path.join(CUSTOM_DIR, ".u2net"))   # rembg 模型放這裡，重建容器也不用重下載

_jobs: "queue.Queue[str]" = queue.Queue()
_status: dict = {}            # id → {"status","progress","error","mode"}
_lock = threading.Lock()
_worker = None


# ───────────── 小工具 ─────────────

def valid_id(avatar_id) -> bool:
    return bool(ID_RE.match(str(avatar_id or "")))


def is_ready(avatar_id) -> bool:
    return valid_id(avatar_id) and os.path.isfile(os.path.join(AVATAR_DIR, avatar_id, "coords.pkl")) \
        and _meta(avatar_id).get("status") == "ready"


def _dir(avatar_id) -> str:
    return os.path.join(CUSTOM_DIR, avatar_id)


def _meta(avatar_id) -> dict:
    try:
        with open(os.path.join(_dir(avatar_id), "meta.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_meta(avatar_id, **kw) -> None:
    m = _meta(avatar_id)
    m.update(kw)
    tmp = os.path.join(_dir(avatar_id), "meta.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(_dir(avatar_id), "meta.json"))


def _set(avatar_id, **kw) -> None:
    with _lock:
        _status.setdefault(avatar_id, {}).update(kw)
    if "status" in kw or "error" in kw or "mode" in kw:
        _save_meta(avatar_id, **{k: v for k, v in kw.items() if k in ("status", "error", "mode")})


def _clean_name(s) -> str:
    s = re.sub(r"[\x00-\x1f【】<>{}\"'\\]", "", str(s or "")).strip()
    return s[:MAX_NAME] or "我的學伴"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _font(size):
    from PIL import ImageFont
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size), True
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size), False
    except TypeError:
        return ImageFont.load_default(), False


# ───────────── 製作流程 ─────────────

_rembg_session = None


def _cutout_rembg(img):
    """用 rembg（u2net_human_seg）把人切出來；沒裝就回傳 None"""
    global _rembg_session
    try:
        from rembg import new_session, remove
    except Exception:
        return None
    try:
        if _rembg_session is None:
            os.makedirs(os.environ["U2NET_HOME"], exist_ok=True)
            _rembg_session = new_session("u2net_human_seg")
        out = remove(img, session=_rembg_session, post_process_mask=True)
        return out.convert("RGBA")
    except Exception as e:
        logger.warning("[gf-custom] rembg 去背失敗：%s", e)
        return None


def _cutout_key(img):
    """證件照這種單色（可有漸層）背景：用上緣和左右兩邊上半部估出背景顏色分布，把跟邊緣相連的背景挖掉；
    背景不夠單純就回傳 None"""
    import cv2
    import numpy as np
    from PIL import Image
    small = img.copy()
    small.thumbnail((900, 900), Image.LANCZOS)
    rgb = np.asarray(small)
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = max(4, min(h, w) // 60)
    ys, xs = np.mgrid[0:h, 0:w]
    sel = np.zeros((h, w), bool)
    sel[:m] = True
    sel[: int(h * 0.55), :m] = True        # 左右只取上半部，下半部常被肩膀擋住
    sel[: int(h * 0.55), -m:] = True
    X = np.stack([np.ones(sel.sum()), xs[sel] / w, ys[sel] / h, (xs[sel] / w) ** 2, (ys[sel] / h) ** 2,
                  xs[sel] * ys[sel] / (w * h)], 1)
    Y = lab[sel]
    keep = np.ones(len(Y), bool)
    for _ in range(3):                      # 去掉頭髮碰到邊緣這種離群點再算一次
        coef, *_ = np.linalg.lstsq(X[keep], Y[keep], rcond=None)
        res = np.linalg.norm(Y - X @ coef, axis=1)
        keep = res < max(6.0, np.percentile(res[keep], 80) * 2)
    spread = np.percentile(res[keep], 90)
    if spread > 10 or keep.mean() < 0.6:   # 邊緣顏色太雜，不是單純背景
        return None
    F = np.stack([np.ones(h * w), xs.ravel() / w, ys.ravel() / h, (xs.ravel() / w) ** 2, (ys.ravel() / h) ** 2,
                  xs.ravel() * ys.ravel() / (w * h)], 1)
    dist = np.linalg.norm(lab.reshape(-1, 3) - F @ coef, axis=1).reshape(h, w)
    t1 = max(7.0, spread * 1.8)
    t2 = t1 + 12.0
    hard = (dist < t2).astype(np.uint8)
    n, labels = cv2.connectedComponents(hard, connectivity=4)
    seeds = set(np.unique(labels[0])) | set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    seeds.discard(0)
    bgmask = np.isin(labels, list(seeds))
    alpha = np.clip((dist - t1) / (t2 - t1), 0, 1).astype(np.float32)
    alpha[~bgmask] = 1.0                    # 沒跟邊緣相連的（白襯衫領子等）一律保留
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)
    if alpha.mean() < 0.12 or alpha.mean() > 0.95:
        return None
    alpha = cv2.resize(alpha, img.size, interpolation=cv2.INTER_LINEAR)
    rgba = np.dstack([np.asarray(img), (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def _cutout(img):
    if CUSTOM_SEG == "off":
        return None, "off"
    if CUSTOM_SEG in ("auto", "rembg"):
        out = _cutout_rembg(img)
        if out is not None:
            return out, "rembg"
    if CUSTOM_SEG == "key":
        out = _cutout_key(img)
        if out is not None:
            return out, "key"
    return None, "none"


def _hex(c, default):
    try:
        c = c.strip().lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default


def _background():
    """背景：有 data/custom/bg.jpg 就用；沒有就用乾淨的淺色牆面（CUSTOM_BG_COLOR，上亮下暗一點點）"""
    from PIL import Image, ImageOps
    if os.path.isfile(CUSTOM_BG):
        try:
            return ImageOps.fit(Image.open(CUSTOM_BG).convert("RGB"), (W, H), Image.LANCZOS)
        except Exception as e:
            logger.warning("[gf-custom] 背景圖讀不到：%s", e)
    import numpy as np
    base = np.array(_hex(os.getenv("CUSTOM_BG_COLOR", "#E8E3DE"), (232, 227, 222)), np.float32)
    t = (np.arange(H, dtype=np.float32) / H)[:, None, None]
    img = base * (1.03 - 0.08 * t)                         # 很淡的上下漸層，不會像一張平板色卡
    img = np.broadcast_to(img, (H, W, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


_sr_net = None


def _sr_load():
    """Real-ESRGAN realesr-general-x4v3（SRVGGNetCompact，約 5 MB）；第一次用時下載到 data/custom/.models"""
    global _sr_net
    if _sr_net is not None:
        return _sr_net
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SRVGG(nn.Module):
        def __init__(self, feat=64, nconv=32, scale=4):
            super().__init__()
            self.scale = scale
            body = [nn.Conv2d(3, feat, 3, 1, 1), nn.PReLU(feat)]
            for _ in range(nconv):
                body += [nn.Conv2d(feat, feat, 3, 1, 1), nn.PReLU(feat)]
            body += [nn.Conv2d(feat, 3 * scale * scale, 3, 1, 1)]
            self.body = nn.ModuleList(body)
            self.up = nn.PixelShuffle(scale)

        def forward(self, x):
            out = x
            for m in self.body:
                out = m(out)
            return self.up(out) + F.interpolate(x, scale_factor=self.scale, mode="nearest")

    if not os.path.isfile(CUSTOM_SR_MODEL):
        import urllib.request
        os.makedirs(os.path.dirname(CUSTOM_SR_MODEL), exist_ok=True)
        tmp = CUSTOM_SR_MODEL + ".part"
        logger.info("[gf-custom] 下載畫質模型 %s", CUSTOM_SR_URL)
        try:
            with urllib.request.urlopen(CUSTOM_SR_URL, timeout=30) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(tmp, CUSTOM_SR_MODEL)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError("下載不到畫質模型；請在主機下載後用 docker cp 放到 /app/data/custom/.models/")
    sd = torch.load(CUSTOM_SR_MODEL, map_location="cpu")
    sd = sd.get("params_ema") or sd.get("params") or sd
    net = SRVGG()
    net.load_state_dict(sd, strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _sr_net = (net.eval().to(dev), dev)
    return _sr_net


def _upscale(img, need: float):
    """照片解析度不夠（放進畫布時要放大，或接近原尺寸）就處理：Real-ESRGAN 放大 4 倍再縮回需要的大小（最多 2.5 倍）"""
    from PIL import Image
    if not CUSTOM_SR or need < 0.85:
        return img, False
    try:
        import numpy as np
        import torch
        net, dev = _sr_load()
        src = img
        if max(src.size) > 1100:                       # 太大的圖先縮小，避免記憶體不夠
            src = src.copy()
            src.thumbnail((1100, 1100), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(src).astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            y = net(x).clamp(0, 1)
        out = Image.fromarray((y[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))
        f = min(max(need * 1.15, 1.0), 2.5)
        size = (round(img.width * f), round(img.height * f))
        out = out.resize(size, Image.LANCZOS)
        logger.info("[gf-custom] 照片放大 %.2f 倍（Real-ESRGAN，%s）：%s → %s", f, dev, img.size, out.size)
        return out, True
    except Exception as e:
        logger.warning("[gf-custom] 畫質放大失敗，用原圖：%s", e)
        return img, False


def _decontaminate(cut, src=None):
    """去掉頭髮邊緣殘留的原背景顏色（例如藍底證件照的藍邊）：
       1. pymatting 估計真正的前景顏色
       2. 邊緣半透明的那圈收緊一點，去掉外圍的淡淡光暈
       3. 邊緣一圈改用內側（頭髮本身）的顏色往外延伸，原背景色就不會留在邊上
       4. 靠近邊緣的一段，把跟原背景同方向的色偏拿掉（藍底就去藍）"""
    import cv2
    import numpy as np
    from PIL import Image
    rgba = np.asarray(cut).astype(np.float32) / 255.
    rgb, a = rgba[..., :3], rgba[..., 3]
    try:
        from pymatting import estimate_foreground_ml
        rgb = estimate_foreground_ml(rgb.astype(np.float64), a.astype(np.float64)).astype(np.float32)
    except Exception as e:
        logger.warning("[gf-custom] pymatting 去色失敗，只做邊緣延伸：%s", e)
    a = np.clip((a - 0.18) / 0.82, 0, 1)
    k = max(3, int(round(min(a.shape) / 300)))           # 邊緣寬度跟著照片大小
    inner = cv2.erode((a > 0.98).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(np.float32)
    sig = 2.5 * k
    num = cv2.GaussianBlur(rgb * inner[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(inner, (0, 0), sig)[..., None]
    ext = num / np.maximum(den, 1e-4)
    w = 1 - cv2.GaussianBlur(inner, (0, 0), k * 0.7)       # 越靠外越用延伸的顏色
    w = np.clip(w * 1.3, 0, 1)[..., None] * (den > 1e-3)
    rgb = rgb * (1 - w) + ext * w
    if src is not None:
        try:
            s_rgb = np.asarray(src).astype(np.float32) / 255.
            a0 = rgba[..., 3]
            bgpix = s_rgb[a0 < 0.03]
            if len(bgpix) > 500:
                lab_bg = cv2.cvtColor(np.median(bgpix, 0).reshape(1, 1, 3).astype(np.float32), cv2.COLOR_RGB2LAB)[0, 0]
                cvec = lab_bg[1:]
                cn = float(np.hypot(*cvec))
                if cn > 8:                                   # 背景有明顯顏色（藍、紅、綠底）才做
                    cdir = cvec / cn
                    band = max(6, int(min(a.shape) * 0.035))
                    near = 1 - cv2.erode((a > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=band).astype(np.float32)
                    near = cv2.GaussianBlur(near, (0, 0), band / 2)
                    lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
                    proj = lab[..., 1] * cdir[0] + lab[..., 2] * cdir[1]
                    rm = np.clip(proj, 0, None) * np.clip(near * 1.2, 0, 1)
                    lab[..., 1] -= rm * cdir[0]
                    lab[..., 2] -= rm * cdir[1]
                    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception as e:
            logger.warning("[gf-custom] 去背景色偏失敗，略過：%s", e)
    out = np.dstack([np.clip(rgb, 0, 1), a])
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


def _fade_cut_sides(cut, bbox) -> None:
    """證件照的肩膀、手臂常被照片左右邊切成一直線：碰到邊的那一側只做很窄的柔邊，
    手臂幾乎完整保留（CUSTOM_SIDE_FADE，人寬度的比例，預設 0.015；設 0 完全不淡出）"""
    import numpy as np
    from PIL import Image
    a = np.asarray(cut.getchannel("A")).astype(np.float32)
    h, w = a.shape
    frac = float(os.getenv("CUSTOM_SIDE_FADE", "0.015"))
    if frac <= 0:
        return
    ramp = max(3, int((bbox[2] - bbox[0]) * frac))
    xs = np.arange(w, dtype=np.float32)
    f = np.ones(w, np.float32)
    if a[:, :2].max() > 40:
        f = np.minimum(f, np.clip(xs / ramp, 0, 1))
    if a[:, -2:].max() > 40:
        f = np.minimum(f, np.clip((w - 1 - xs) / ramp, 0, 1))
    if f.min() < 1:
        cut.putalpha(Image.fromarray((a * f[None, :]).astype(np.uint8)))


def _prepare(avatar_id, raw: bytes) -> str:
    """照片 → 1280×720 畫布：把人去背，放在房間背景下方中間；去背失敗才用「置中＋模糊背景」"""
    from PIL import Image, ImageFilter, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    if min(img.size) < 256:
        raise ValueError("照片太小了，請用至少 512 像素寬的照片")
    img, sr = _upscale(img, min(H * 0.94 / img.height, W * 0.9 / img.width))
    cut, how = _cutout(img)
    if cut is not None and how == "rembg":
        cut = _decontaminate(cut, img)
    if cut is not None:
        bbox = cut.getchannel("A").point(lambda v: 255 if v > 40 else 0).getbbox()
        if not bbox:
            cut = None
    if cut is not None:
        _fade_cut_sides(cut, bbox)
        person = cut.crop(bbox)
        at_bottom = bbox[3] >= img.height - max(3, img.height // 50)   # 證件照、半身照：身體被照片下緣切掉
        target_h = H * (0.94 if at_bottom else 0.86)
        scale = min(target_h / person.height, W * 0.9 / person.width)
        person = person.resize((max(1, round(person.width * scale)), max(1, round(person.height * scale))), Image.LANCZOS)
        bg = _background()
        px = (W - person.width) // 2
        py = H - person.height if at_bottom else (H - person.height) // 2 + int(H * 0.04)
        bg = bg.convert("RGBA")
        bg.alpha_composite(person, (px, py))
        bg = bg.convert("RGB")
        logger.info("[gf-custom] %s 去背成功（%s）", avatar_id, how)
    else:
        bg = ImageOps.fit(img, (W, H)).filter(ImageFilter.GaussianBlur(28))
        bg = Image.blend(bg, Image.new("RGB", (W, H), (20, 18, 26)), 0.35)
        fg = img.copy()
        fg.thumbnail((W, H), Image.LANCZOS)
        bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
        logger.info("[gf-custom] %s 沒有去背（%s），用模糊背景", avatar_id, how)
    d = _dir(avatar_id)
    canvas = os.path.join(d, "canvas.jpg")
    bg.save(canvas, quality=94)
    os.chmod(canvas, 0o644)
    tw = int(H * 0.8)
    th = bg.crop(((W - tw) // 2, 0, (W + tw) // 2, H))
    th.thumbnail((360, 360))
    th.save(os.path.join(d, "thumb.jpg"), quality=86)
    return canvas


def _driving_video() -> str | None:
    """動作範本：LP_DRIVING；沒有就用小晴的待機畫面做一支（前 10 秒）"""
    if os.path.isfile(LP_DRIVING):
        return LP_DRIVING
    import cv2
    try:
        sys.path.insert(0, os.getcwd())
        main = sys.modules.get("__main__")
        default_id = getattr(getattr(main, "opt", None), "avatar_id", None) or "xiaoqing"
    except Exception:
        default_id = "xiaoqing"
    frames = sorted(glob.glob(os.path.join(AVATAR_DIR, default_id, "full_imgs", "*.png")))[:FPS * 10]
    if not frames:
        return None
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(LP_DRIVING), exist_ok=True)
    vw = cv2.VideoWriter(LP_DRIVING, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in frames:
        vw.write(cv2.imread(f))
    vw.release()
    logger.info("[gf-custom] 用 %s 的待機畫面做了動作範本 %s（%d 格）", default_id, LP_DRIVING, len(frames))
    return LP_DRIVING


def _static_video(avatar_id, canvas: str) -> str:
    import cv2
    img = cv2.imread(canvas)
    out = os.path.join(_dir(avatar_id), "idle.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (img.shape[1], img.shape[0]))
    for _ in range(FPS * 4):
        vw.write(img)
    vw.release()
    return out


def _animate(avatar_id, canvas: str) -> tuple[str, str]:
    """回傳 (影片路徑, 模式)；LivePortrait 失敗就退回靜態版"""
    if LP_CMD:
        try:
            driving = _driving_video()
            if not driving:
                raise RuntimeError("找不到動作範本影片")
            outdir = os.path.join(_dir(avatar_id), "lp")
            os.makedirs(outdir, exist_ok=True)
            cmd = LP_CMD.format(source=shlex.quote(canvas), driving=shlex.quote(driving), outdir=shlex.quote(outdir))
            logger.info("[gf-custom] LivePortrait：%s", cmd)
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1200)
            vids = [v for v in glob.glob(os.path.join(outdir, "**", "*.mp4"), recursive=True) if "concat" not in v]
            if p.returncode != 0 or not vids:
                raise RuntimeError((p.stderr or p.stdout or "沒有輸出影片")[-400:])
            return max(vids, key=os.path.getmtime), "animated"
        except Exception as e:
            logger.warning("[gf-custom] LivePortrait 失敗，改用靜態版：%s", e)
            _set(avatar_id, error=f"動態版失敗，先用靜態版：{str(e)[-120:]}")
    elif worker_alive():
        try:
            return _animate_by_worker(avatar_id), "animated"
        except Exception as e:
            try:
                os.remove(os.path.join(_dir(avatar_id), "lp_request"))   # 小幫手之後啟動也不要再做
            except OSError:
                pass
            logger.warning("[gf-custom] 小幫手做待機影片失敗，改用靜態版：%s", e)
            _set(avatar_id, error=f"動態版失敗，先用靜態版：{str(e)[-120:]}")
    return _static_video(avatar_id, canvas), "static"


def worker_alive() -> bool:
    try:
        return time.time() - os.path.getmtime(LP_HEARTBEAT) < 30
    except OSError:
        return False


def _animate_by_worker(avatar_id) -> str:
    """放一個 lp_request 標記，等主機上的小幫手做完 idle_lp.mp4"""
    d = _dir(avatar_id)
    for f in ("lp_done", "lp_failed", "idle_lp.mp4"):
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass
    open(os.path.join(d, "lp_request"), "w").close()
    os.chmod(os.path.join(d, "lp_request"), 0o666)
    t0 = time.time()
    while time.time() - t0 < LP_TIMEOUT:
        if os.path.exists(os.path.join(d, "lp_done")) and os.path.getsize(os.path.join(d, "idle_lp.mp4")) > 0:
            return os.path.join(d, "idle_lp.mp4")
        if os.path.exists(os.path.join(d, "lp_failed")):
            try:
                tail = open(os.path.join(d, "lp.log"), encoding="utf-8", errors="replace").read()[-300:]
            except OSError:
                tail = ""
            lines = tail.strip().splitlines()
            raise RuntimeError("照片裡找不到清楚的臉" if "找不到臉" in tail else (lines[-1] if lines else "LivePortrait 失敗"))
        if not os.path.exists(os.path.join(d, "lp_running")) and not worker_alive():
            raise RuntimeError("小幫手停止了")
        _set(avatar_id, progress=min(38, 8 + int((time.time() - t0) / 4)))
        time.sleep(1)
    raise RuntimeError("等太久了")


def _watermark(avatar_id) -> None:
    """人物畫面左上角印「AI 生成」（臉部是另外貼上的，不受影響）"""
    from PIL import Image, ImageDraw
    font, cjk = _font(26)
    label = "AI 生成" if cjk else "AI GENERATED"
    for f in glob.glob(os.path.join(AVATAR_DIR, avatar_id, "full_imgs", "*.png")):
        im = Image.open(f).convert("RGB")
        dr = ImageDraw.Draw(im, "RGBA")
        x, y = 18, 16
        box = dr.textbbox((x, y), label, font=font)
        dr.rounded_rectangle((box[0] - 10, box[1] - 6, box[2] + 10, box[3] + 6), 10, fill=(0, 0, 0, 120))
        dr.text((x, y), label, font=font, fill=(255, 255, 255, 230))
        im.save(f)


def _check_face(avatar_id) -> None:
    """generate_avatar 找不到臉時會用整張圖當臉框，這裡把它擋下來"""
    with open(os.path.join(AVATAR_DIR, avatar_id, "coords.pkl"), "rb") as f:
        coords = pickle.load(f)
    y1, y2, x1, x2 = coords[0]
    if (y2 - y1) >= H * 0.95 and (x2 - x1) >= W * 0.95:
        raise RuntimeError("照片裡找不到清楚的臉，請換一張正面、臉部清楚的照片")


def _build(avatar_id) -> None:
    d = _dir(avatar_id)
    try:
        _set(avatar_id, status="animating", progress=5)
        with open(os.path.join(d, "photo.bin"), "rb") as f:
            raw = f.read()
        canvas = _prepare(avatar_id, raw)
        video, mode = _animate(avatar_id, canvas)
        _set(avatar_id, status="building", progress=40, mode=mode)
        from avatars.wav2lip.genavatar import generate_avatar
        generate_avatar(video_path=video, avatar_id=avatar_id, save_path=AVATAR_DIR, img_size=IMG_SIZE,
                        progress_callback=lambda p: _set(avatar_id, progress=40 + int(p * 0.55)))
        _check_face(avatar_id)
        if CUSTOM_WATERMARK:
            _watermark(avatar_id)
        _set(avatar_id, status="ready", progress=100)
        logger.info("[gf-custom] %s 完成（%s）", avatar_id, mode)
    except Exception as e:
        logger.exception("[gf-custom] %s 製作失敗", avatar_id)
        shutil.rmtree(os.path.join(AVATAR_DIR, avatar_id), ignore_errors=True)
        _set(avatar_id, status="failed", progress=0, error=str(e)[:200])
    finally:
        try:
            os.remove(os.path.join(d, "photo.bin"))        # 原始照片做完就刪，只留畫布和縮圖
        except OSError:
            pass


def _run_worker():
    while True:
        avatar_id = _jobs.get()
        if os.path.isdir(_dir(avatar_id)):
            _build(avatar_id)


def _ensure_worker():
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_run_worker, daemon=True, name="gf-custom")
        _worker.start()


# ───────────── 給 app.py / 其他模組用 ─────────────

def after_session(avatar_session, avatar_id) -> None:
    """自訂人物的連線：標記、套上名字、預設就是學伴人設"""
    if not valid_id(avatar_id):
        return
    avatar_session._gf_custom = avatar_id
    avatar_session._gf_name = _meta(avatar_id).get("name") or "我的學伴"
    try:
        import llm
        pid = next((k for k, v in llm._personas()["personas"].items() if v.get("board")), None)
        if pid:
            llm.set_persona(avatar_session, pid)
    except Exception:
        logger.exception("[gf-custom] 設定學伴人設失敗")


def safe_avatar_id(requested, default_id):
    """/offer 帶來的人物名稱只接受預設人物或做好的自訂學伴，避免讀到別的資料夾"""
    if not requested or requested == default_id:
        return default_id
    if is_ready(requested):
        return requested
    logger.warning("[gf-custom] 拒絕不存在或不合法的人物：%r", str(requested)[:60])
    return default_id


# ───────────── HTTP ─────────────

async def create(request):
    try:
        reader = await request.multipart()
        fields, raw = {}, None
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "photo":
                buf = bytearray()
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    buf += chunk
                    if len(buf) > MAX_PHOTO:
                        return web.json_response({"ok": False, "msg": "照片太大（上限 12 MB）"}, status=400)
                raw = bytes(buf)
            else:
                fields[part.name] = (await part.text())[:200]
        if not raw:
            return web.json_response({"ok": False, "msg": "請選一張照片"}, status=400)
        if not all(fields.get(k) in ("1", "true", "on") for k in ("consent_self", "consent_adult", "consent_public")):
            return web.json_response({"ok": False, "msg": "請先勾選三個同意事項"}, status=400)
        try:
            from PIL import Image
            Image.open(io.BytesIO(raw)).verify()
        except Exception:
            return web.json_response({"ok": False, "msg": "看不懂這個檔案，請上傳 JPG 或 PNG 照片"}, status=400)

        avatar_id = "custom_" + secrets.token_hex(8)
        key = secrets.token_urlsafe(18)
        d = _dir(avatar_id)
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o777)                              # 主機上的小幫手（一般使用者）要能寫進來
        with open(os.path.join(d, "photo.bin"), "wb") as f:
            f.write(raw)
        _save_meta(avatar_id, name=_clean_name(fields.get("name")), key=_hash(key), created=int(time.time()),
                   consent={"self": True, "adult": True, "not_public_figure": True, "at": int(time.time())},
                   status="queued", mode="")
        _set(avatar_id, status="queued", progress=0)
        _ensure_worker()
        _jobs.put(avatar_id)
        logger.info("[gf-custom] 新的自訂學伴 %s（排隊 %d）", avatar_id, _jobs.qsize())
        return web.json_response({"ok": True, "id": avatar_id, "key": key, "queue": _jobs.qsize()})
    except Exception as e:
        logger.exception("[gf-custom] /custom/create exception:")
        return web.json_response({"ok": False, "msg": str(e)}, status=500)


async def status(request):
    avatar_id = request.query.get("id", "")
    if not valid_id(avatar_id) or not os.path.isdir(_dir(avatar_id)):
        return web.json_response({"ok": False, "status": "missing"}, status=404)
    m = _meta(avatar_id)
    with _lock:
        live = dict(_status.get(avatar_id, {}))
    st = live.get("status") or m.get("status")
    if st in ("queued", "animating", "building") and avatar_id not in _status:
        st = "failed"                                   # 伺服器重開過，沒做完的不會繼續
    return web.json_response({"ok": True, "status": st, "progress": live.get("progress", 100 if st == "ready" else 0),
                              "name": m.get("name"), "mode": m.get("mode") or live.get("mode"),
                              "error": live.get("error") or m.get("error")})


async def thumb(request):
    avatar_id = request.query.get("id", "")
    p = os.path.join(_dir(avatar_id), "thumb.jpg")
    if not valid_id(avatar_id) or not os.path.isfile(p):
        return web.Response(status=404)
    return web.FileResponse(p, headers={"Cache-Control": "private, max-age=3600"})


async def delete(request):
    try:
        params = await request.json()
        avatar_id, key = str(params.get("id", "")), str(params.get("key", ""))
        if not valid_id(avatar_id) or not os.path.isdir(_dir(avatar_id)):
            return web.json_response({"ok": False, "msg": "找不到這個學伴"}, status=404)
        if not key or _hash(key) != _meta(avatar_id).get("key"):
            return web.json_response({"ok": False, "msg": "只有建立的人可以刪除"}, status=403)
        shutil.rmtree(os.path.join(AVATAR_DIR, avatar_id), ignore_errors=True)
        shutil.rmtree(_dir(avatar_id), ignore_errors=True)
        with _lock:
            _status.pop(avatar_id, None)
        main = sys.modules.get("__main__")
        cache = getattr(main, "global_avatars", None)
        if isinstance(cache, dict):
            cache.pop(avatar_id, None)                  # 從記憶體移除（正在用的連線不受影響）
        logger.info("[gf-custom] 已刪除 %s", avatar_id)
        return web.json_response({"ok": True})
    except Exception as e:
        logger.exception("[gf-custom] /custom/delete exception:")
        return web.json_response({"ok": False, "msg": str(e)}, status=500)


PATHS = {"/custom/create", "/custom/status", "/custom/thumb", "/custom/delete"}


def setup(app):
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    try:
        os.chmod(CUSTOM_DIR, 0o777)
    except OSError:
        pass
    app.router.add_post("/custom/create", create)
    app.router.add_get("/custom/status", status)
    app.router.add_get("/custom/thumb", thumb)
    app.router.add_post("/custom/delete", delete)
    logger.info("[gf-custom] 自訂學伴開啟（LivePortrait：%s）",
                "容器內指令" if LP_CMD else "主機小幫手（tools/lp_worker.sh），沒開就用靜態版")
