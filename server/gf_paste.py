"""說話時的嘴型貼回：只換下半臉、邊緣羽化、稍微銳化。

LiveTalking 原本的做法是把 wav2lip 產生的 256×256 整張臉直接蓋回去（cv2.resize 放大成臉框大小），
所以一開口整張臉（連眼睛）都會變糊，臉框邊緣也看得到一條方框。
wav2lip 只會改下半臉（輸入時上半臉是參考、下半臉被遮住），上半臉用原圖就好：
  - 上半臉保留原本清楚的畫面，從臉框約 45%～62% 高度慢慢換成嘴型畫面
  - 左右、下緣羽化，看不出方框
  - 嘴型畫面用 Lanczos 放大，再做輕微銳化

環境變數：
  GF_PASTE          soft（預設）或 off（回到原本整張蓋上）
  GF_PASTE_SHARPEN  銳化強度，預設 0.6；0 表示不銳化
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from utils.logger import logger

MODE = os.getenv("GF_PASTE", "soft").strip().lower()
SHARPEN = float(os.getenv("GF_PASTE_SHARPEN", "0.6"))
TOP0, TOP1 = 0.45, 0.62        # 這段高度從原圖漸變到嘴型畫面
SIDE = 0.10                    # 左右羽化寬度（臉框寬的比例）
BOTTOM = 0.06                  # 下緣羽化

_masks: dict = {}


def _mask(h: int, w: int) -> np.ndarray:
    m = _masks.get((h, w))
    if m is None:
        ys = (np.arange(h, dtype=np.float32) + 0.5) / h
        xs = (np.arange(w, dtype=np.float32) + 0.5) / w
        v = np.clip((ys - TOP0) / (TOP1 - TOP0), 0, 1) * np.clip((1 - ys) / BOTTOM, 0, 1)
        u = np.clip(xs / SIDE, 0, 1) * np.clip((1 - xs) / SIDE, 0, 1)
        m = (v[:, None] * u[None, :])
        m = m * m * (3 - 2 * m)            # smoothstep，過渡更柔
        m = np.ascontiguousarray(m[..., None].astype(np.float32))
        if len(_masks) > 64:
            _masks.clear()
        _masks[(h, w)] = m
    return m


def _soft_paste(self, pred_frame, idx: int):
    bbox = self.coord_list_cycle[idx]
    combine_frame = self.frame_list_cycle[idx].copy()
    y1, y2, x1, x2 = bbox
    h, w = y2 - y1, x2 - x1
    if h <= 4 or w <= 4:
        return combine_frame
    res = cv2.resize(pred_frame.astype(np.uint8), (w, h), interpolation=cv2.INTER_LANCZOS4)
    if SHARPEN > 0:
        blur = cv2.GaussianBlur(res, (0, 0), max(0.8, w / 256 * 0.9))
        res = cv2.addWeighted(res, 1 + SHARPEN, blur, -SHARPEN, 0)
    m = _mask(h, w)
    orig = combine_frame[y1:y2, x1:x2].astype(np.float32)
    combine_frame[y1:y2, x1:x2] = (orig + (res.astype(np.float32) - orig) * m).astype(np.uint8)
    return combine_frame


def patch() -> None:
    if MODE == "off":
        logger.info("[gf-paste] 使用 LiveTalking 原本的貼回方式")
        return
    try:
        from avatars.wav2lip_avatar import LipReal
    except Exception as e:
        logger.warning(f"[gf-paste] 找不到 wav2lip 類別，略過：{e}")
        return
    LipReal.paste_back_frame = _soft_paste
    logger.info(f"[gf-paste] 說話時只換下半臉、邊緣羽化（銳化 {SHARPEN}）")
