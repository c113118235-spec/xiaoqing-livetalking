"""小晴的姿勢與動作（坐 ↔ 站、揮手…）—— 放在 server/gf_pose.py，由 gf_routes.py 載入

設定檔 data/actions/poses.json（範例）：
{
  "default_pose": "sit",
  "poses":   {"sit": "xiaoqing", "stand": "xiaoqing_stand"},
  "idle":    {"actions": ["look_window"], "min_sec": 25, "max_sec": 60},
  "actions": {
    "stand": {"frames": "stand_up",              "from": "sit",   "to": "stand"},
    "sit":   {"frames": "stand_up", "reverse": true, "from": "stand", "to": "sit"},
    "wave":  {"frames": "wave_sit", "pingpong": true, "from": "sit",  "to": "sit"}
  }
}
- poses：每個姿勢對應一個 LiveTalking 人物資料夾（data/avatars/<id>），各自有全身畫面、臉、臉框，說話時照常對嘴
- actions：動作影片的畫面放在 data/actions/<frames>/*.png（25 fps、與人物同尺寸）
    reverse：倒著播（站起來倒播 = 坐下）；pingpong：播完再倒播回第一格（揮手後回到原姿勢）

原理（不改 LiveTalking 原始碼）：
1. 把每個連線的 frame/face/coord 清單換成「姿勢視圖」，換姿勢時只換視圖裡的資料，並讓待機從第 0 格開始
2. 包住 output.push_video_frame：做動作時用動作影片的畫面取代，頭尾各淡入淡出幾格
3. 她說話時不開始動作，會等這句說完；llm.py 在動作進行中會先等動作做完再送下一句
"""
import glob
import json
import os
import random
import threading
import time

import cv2
import numpy as np

from utils.logger import logger

CONFIG = os.getenv("GF_POSE_CONFIG", "data/actions/poses.json")
FADE = 6                      # 動作開始時淡入的格數
WATERMARK = "LiveTalking"     # LiveTalking 授權要求的標示

_lock = threading.Lock()
_cfg = None                   # poses.json 內容
_avatars = {}                 # 姿勢名稱 → (frames, faces, coords)
_clips = {}                   # 動作名稱 → [frame, ...]


def _mirror(n, i):
    if n <= 0:
        return 0
    i = i % (2 * n)
    return i if i < n else 2 * n - 1 - i


def _load_cfg():
    global _cfg
    if _cfg is None:
        try:
            with open(CONFIG, encoding="utf-8") as f:
                _cfg = json.load(f)
        except OSError:
            _cfg = {}
        except Exception:
            logger.exception("[gf-pose] poses.json 讀取失敗")
            _cfg = {}
    return _cfg


def enabled():
    return bool(_load_cfg().get("poses"))


def _avatar(pose, default_avatar):
    """姿勢 → (frames, faces, coords)；預設姿勢直接用 LiveTalking 已載入的人物"""
    with _lock:
        if pose in _avatars:
            return _avatars[pose]
        cfg = _load_cfg()
        avatar_id = cfg.get("poses", {}).get(pose)
        if pose == cfg.get("default_pose") or avatar_id is None:
            data = default_avatar
        else:
            from avatars.wav2lip_avatar import load_avatar
            t = time.time()
            data = load_avatar(avatar_id)
            logger.info(f"[gf-pose] 載入姿勢 {pose}（{avatar_id}，{len(data[0])} 格，{time.time() - t:.1f}s）")
        _avatars[pose] = data
        return data


def _clip(name):
    """動作名稱 → 要播的畫面清單"""
    with _lock:
        if name in _clips:
            return _clips[name]
        a = _load_cfg().get("actions", {}).get(name)
        if not a:
            return None
        files = sorted(glob.glob(os.path.join("data/actions", a["frames"], "*.[jpJP][pnPN]*[gG]")))
        if not files:
            logger.warning(f"[gf-pose] 動作 {name} 找不到畫面：data/actions/{a['frames']}")
            return None
        frames = [cv2.imread(f) for f in files]
        if a.get("reverse"):
            frames = frames[::-1]
        if a.get("pingpong"):
            frames = frames + frames[-2::-1]
        _clips[name] = frames
        logger.info(f"[gf-pose] 載入動作 {name}：{len(frames)} 格（{len(frames) / 25:.1f} 秒）")
        return frames


class _View:
    """假裝是一個超長的清單：LiveTalking 拿到的索引不經過 mirror，由這裡依目前姿勢換算"""
    BIG = 10 ** 9

    def __init__(self, ctl, k):
        self.ctl, self.k = ctl, k

    def __len__(self):
        return self.BIG

    def __getitem__(self, raw):
        data = self.ctl.data[self.k]
        if self.k == 0:                       # 全身畫面：記下輸出端看到的位置
            self.ctl.last_raw = raw
        return data[_mirror(len(data), raw - self.ctl.offset)]


class PoseController:
    def __init__(self, session, default_avatar):
        self.s = session
        self.default_avatar = default_avatar
        self.pose = _load_cfg().get("default_pose", "sit")
        self.data = _avatar(self.pose, default_avatar)
        self.offset = 0
        self.last_raw = 0
        self.playing = None          # 正在播的動作畫面
        self.pos = 0
        self.pending = None          # 等她說完話才開始的動作
        self.last_frame = None
        # 待機時自動做的小動作（poses.json 的 "idle"）
        self.idle_frames = 0         # 連續待機（沒說話、沒動作）的格數
        self.idle_next = self._next_idle()
        # 換上姿勢視圖
        session.frame_list_cycle = _View(self, 0)
        session.face_list_cycle = _View(self, 1)
        session.coord_list_cycle = _View(self, 2)
        # 包住畫面輸出
        out = session.output
        self._push = out.push_video_frame
        out.push_video_frame = self._push_video_frame

    # ---- 對外 ----
    def busy(self):
        return self.playing is not None or self.pending is not None

    def request(self, name):
        a = _load_cfg().get("actions", {}).get(name)
        if not a:
            return False, f"沒有這個動作：{name}"
        if a.get("from") and a["from"] != self.pose:
            return False, f"現在是「{self.pose}」，不能做「{name}」"
        if self.busy():
            return False, "正在做別的動作"
        if _clip(name) is None:
            return False, f"動作 {name} 的畫面還沒準備好"
        self.pending = name
        return True, "ok"

    # ---- 內部 ----
    def _next_idle(self):
        idle = _load_cfg().get("idle") or {}
        lo, hi = idle.get("min_sec", 25), idle.get("max_sec", 60)
        return int(random.uniform(lo, hi) * 25)

    def _maybe_idle_action(self):
        """待機夠久就隨機挑一個 idle 動作（只挑目前姿勢能做的）"""
        if self.s.speaking or self.busy():
            self.idle_frames = 0
            return
        self.idle_frames += 1
        if self.idle_frames < self.idle_next:
            return
        self.idle_frames, self.idle_next = 0, self._next_idle()
        cfg = _load_cfg()
        names = [n for n in (cfg.get("idle") or {}).get("actions", [])
                 if cfg.get("actions", {}).get(n, {}).get("from", self.pose) == self.pose]
        if names:
            ok, why = self.request(random.choice(names))
            logger.info(f"[gf-pose] 待機小動作：{why}")

    def _start(self, name):
        a = _load_cfg()["actions"][name]
        frames = _clip(name)
        to = a.get("to", self.pose)
        if to != self.pose:
            self.data = _avatar(to, self.default_avatar)
            self.pose = to
        # 動作播完時，待機剛好從第 0 格開始（站起來最後一格 = 站姿第 0 格）
        self.offset = self.last_raw + len(frames)
        self.playing, self.pos = frames, 0
        logger.info(f"[gf-pose] 開始動作 {name}（{len(frames)} 格），之後姿勢：{self.pose}")

    def _push_video_frame(self, frame):
        self._maybe_idle_action()
        if self.playing is None and self.pending and not self.s.speaking:
            name, self.pending = self.pending, None
            self._start(name)
        if self.playing is not None:
            f = self.playing[self.pos].copy()
            if self.pos < FADE and self.last_frame is not None and self.last_frame.shape == f.shape:
                a = (self.pos + 1) / (FADE + 1)
                f = cv2.addWeighted(self.last_frame, 1 - a, f, a, 0)
            cv2.putText(f, WATERMARK, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
            self.pos += 1
            if self.pos >= len(self.playing):
                self.playing = None
            frame = f
        else:
            self.last_frame = frame
        self._push(frame)


def attach(session, default_avatar):
    """在每個新連線建立後呼叫；沒有 poses.json 就什麼都不做"""
    if not enabled() or hasattr(session, "_gf_pose"):
        return
    try:
        session._gf_pose = PoseController(session, default_avatar)
    except Exception:
        logger.exception("[gf-pose] 初始化失敗，這個連線不使用姿勢功能")


def actions():
    cfg = _load_cfg()
    return {"pose_default": cfg.get("default_pose"),
            "idle": (cfg.get("idle") or {}).get("actions", []),
            "actions": {k: {"from": v.get("from"), "to": v.get("to")} for k, v in cfg.get("actions", {}).items()}}


def patch_livetalking():
    """包住 LipReal.__init__，每個連線建立時自動掛上姿勢控制"""
    if not enabled():
        logger.info("[gf-pose] 沒有 data/actions/poses.json，姿勢功能關閉")
        return
    from avatars.wav2lip_avatar import LipReal
    if getattr(LipReal, "_gf_pose_patched", False):
        return
    orig = LipReal.__init__

    def init(self, opt, model, avatar):
        orig(self, opt, model, avatar)
        attach(self, avatar)

    LipReal.__init__ = init
    LipReal._gf_pose_patched = True
    logger.info(f"[gf-pose] 姿勢功能開啟：{list(_load_cfg().get('poses', {}))}，動作：{list(_load_cfg().get('actions', {}))}")
