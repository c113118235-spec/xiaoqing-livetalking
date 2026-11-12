"""聲音：開始畫面的聲音清單、試聽，以及「用自己的聲音」（現場錄音克隆）。

聲音放在 data/voices/<id>/：
  neutral.wav + neutral.txt   參考錄音和逐字稿（IndexTTS / BreezyVoice 都讀這裡）
  meta.json                   {"name": "女聲 2", "gender": "female"}；自己錄的另有 owner（金鑰雜湊）、consent
id 規則：內建聲音 female、male、f2…、m2…（任意小寫英數），使用者自己錄的是 u_<16 碼>，只有錄的人看得到。

API（都在通關碼後面）：
  GET  /tts/voices                    內建聲音清單（不含別人錄的）；?mine=u_xxx,u_yyy 一起回傳自己錄的
  GET  /tts/preview?voice=<id>        試聽一句（第一次產生後快取）
  POST /voice/create   multipart: audio（瀏覽器錄的 webm/mp4/wav）、name、consent_self=1、consent_use=1
                       → {"ok", "id", "key"}；key 只回這一次，刪除要用
  POST /voice/delete   {"id", "key"}

只接受網頁上「現場錄音」：避免拿名人或別人的錄音來克隆（網頁不提供上傳檔案）。
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import time

import aiohttp
from aiohttp import web

from utils.logger import logger

VOICES_DIR = os.path.abspath(os.getenv("VOICES_DIR", os.path.join("data", "voices")))
TW_TTS_URLS = [u.strip() for u in os.getenv(
    "TW_TTS_URLS", "http://127.0.0.1:8007/v1/audio/speech,http://127.0.0.1:8006/v1/audio/speech").split(",") if u.strip()]
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
USER_RE = re.compile(r"^u_[0-9a-f]{16}$")
PREVIEW_TEXT = "嗨，我是你的學伴！今天也一起加油吧，有不懂的地方隨時問我喔。"
RECORD_TEXT = "嗨，我是你的學伴。今天我們一起把作業寫完，有不懂的地方隨時問我，加油！"
IAI_TTS_URL = os.getenv("IAI_TTS_URL", os.getenv("IAI_BASE_URL", "https://www.iai.nkust.edu.tw/aihub/v1").rstrip("/") + "/audio/speech")
IAI_TTS_MODEL = os.getenv("IAI_TTS_MODEL", "Speaker")
IAI_TTS_VOICE = os.getenv("IAI_TTS_VOICE", "default")


def iai_on() -> bool:
    """有 IAI_API_KEY（跟 LLM 同一把）就在開始畫面多一個「iAI 聲音」（雲端，學校 iAI 平台）"""
    return bool(os.getenv("IAI_API_KEY")) and os.getenv("IAI_TTS_ENABLE", "1") == "1"


IAI_ENTRY = {"id": "iai", "name": "iAI 聲音", "gender": "cloud", "mine": False, "order": 0}
MAX_USER_VOICES = int(os.getenv("MAX_USER_VOICES", "200"))
MAX_AUDIO = 8 * 1024 * 1024

_preview_cache: dict = {}           # (id, mtime) → wav bytes
_preview_lock = asyncio.Lock()


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _meta(vid: str) -> dict:
    d = os.path.join(VOICES_DIR, vid)
    m = {}
    try:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        pass
    g = m.get("gender") or ("male" if vid.startswith(("male", "m")) else "female")
    default = {"female": "女聲 1", "male": "男聲 1"}.get(vid, vid)
    return {"id": vid, "name": str(m.get("name") or default)[:12], "gender": g if g in ("female", "male") else "female",
            "mine": vid.startswith("u_"), "order": m.get("order", 99 if vid.startswith("u_") else 50)}


def _list(mine: set) -> list:
    out = []
    if not os.path.isdir(VOICES_DIR):
        return out
    for vid in sorted(os.listdir(VOICES_DIR)):
        if not ID_RE.match(vid) or not os.path.isfile(os.path.join(VOICES_DIR, vid, "neutral.wav")):
            continue
        if vid.startswith("u_") and vid not in mine:
            continue                    # 別人錄的聲音不列出
        out.append(_meta(vid))
    out.sort(key=lambda v: (v["mine"], v["gender"] != "female", v["order"], v["name"]))
    return out


async def _tts_alive() -> str | None:
    for tw in TW_TTS_URLS:
        url = tw.rsplit("/audio/speech", 1)[0] + "/voices"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
                async with s.get(url) as r:
                    if r.status == 200:
                        return tw
        except Exception:
            pass
    return None


async def voices(request):
    mine = {x for x in request.query.get("mine", "").split(",") if USER_RE.match(x)}
    tw = await _tts_alive()
    lst = _list(mine) if tw else []
    if iai_on():
        lst.append(dict(IAI_ENTRY))
    if not lst:
        return web.json_response({"engine": "tts2", "voices": {}, "list": []})
    # 舊版網頁用的格式（voices: {id: [...]}）一起回傳
    return web.json_response({"engine": "breezy", "url": tw, "list": lst,
                              "voices": {v["id"]: ["neutral"] for v in lst}, "record_text": RECORD_TEXT})


def _iai_body(text):
    try:
        from tts.gftts import iai_body       # 跟正式說話用同一組參數（data/tts.json 的 iai）
        return iai_body(text, "wav")
    except Exception:
        return {"model": IAI_TTS_MODEL, "input": text, "voice": IAI_TTS_VOICE}


async def _preview_iai():
    body = _iai_body(PREVIEW_TEXT)
    key = ("iai", json.dumps(body, sort_keys=True, ensure_ascii=False))
    if key in _preview_cache:
        return _preview_cache[key]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        async with s.post(IAI_TTS_URL, headers={"x-litellm-api-key": f"Bearer {os.getenv('IAI_API_KEY', '')}"},
                          json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"iAI {r.status}")
            data = (await r.read(), r.headers.get("Content-Type", "audio/wav").split(";")[0])
    _preview_cache[key] = data
    return data


async def preview(request):
    vid = request.query.get("voice", "")
    if vid == "iai" and iai_on():
        try:
            body, ctype = await _preview_iai()
        except Exception as e:
            return web.json_response({"ok": False, "error": f"iAI 語音沒回應：{e}"}, status=503)
        return web.Response(body=body, content_type=ctype, headers={"Cache-Control": "no-store"})
    if not ID_RE.match(vid) or not os.path.isfile(os.path.join(VOICES_DIR, vid, "neutral.wav")):
        return web.json_response({"ok": False, "error": "沒有這個聲音"}, status=404)
    key = (vid, os.path.getmtime(os.path.join(VOICES_DIR, vid, "neutral.wav")))
    async with _preview_lock:
        wav = _preview_cache.get(key)
        if wav is None:
            last = None
            for tw in TW_TTS_URLS:
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
                        async with s.post(tw, json={"input": PREVIEW_TEXT, "voice": vid, "emotion": "happy"}) as r:
                            if r.status == 200:
                                wav = await r.read()
                                break
                            last = f"{r.status}"
                except Exception as e:
                    last = str(e)
            if wav is None:
                return web.json_response({"ok": False, "error": f"語音服務沒回應：{last}"}, status=503)
            if len(_preview_cache) > 64:
                _preview_cache.clear()
            _preview_cache[key] = wav
    return web.Response(body=wav, content_type="audio/wav", headers={"Cache-Control": "no-store"})


def _decode(raw: bytes):
    """瀏覽器錄的 webm/opus、mp4/aac 或 wav → 24 kHz 單聲道 float32（用 aiortc 自帶的 PyAV 解碼）"""
    import numpy as np
    try:
        import soundfile as sf
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        import av
        chunks = []
        with av.open(io.BytesIO(raw)) as c:
            st = c.streams.audio[0]
            rs = av.AudioResampler(format="flt", layout="mono", rate=24000)
            for frame in c.decode(st):
                for f in rs.resample(frame):
                    chunks.append(f.to_ndarray().reshape(-1))
            for f in rs.resample(None):
                chunks.append(f.to_ndarray().reshape(-1))
        wav, sr = (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), 24000
    if sr != 24000 and len(wav):
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
        sr = 24000
    return wav.astype("float32"), sr


def _clean(wav, sr):
    """去掉頭尾安靜、音量拉到一致；回傳 (wav, 錯誤訊息)"""
    import numpy as np
    if not len(wav):
        return None, "沒有錄到聲音"
    hop = int(sr * 0.02)
    e = np.array([np.sqrt(np.mean(wav[i:i + hop] ** 2)) for i in range(0, max(1, len(wav) - hop), hop)])
    if e.max() < 0.01:
        return None, "聲音太小了，靠近麥克風再錄一次"
    on = np.where(e > e.max() * 0.06)[0]
    a, b = max(0, (on[0] - 5) * hop), min(len(wav), (on[-1] + 8) * hop)
    wav = wav[a:b]
    dur = len(wav) / sr
    if dur < 4:
        return None, f"只錄到 {dur:.1f} 秒，請把整句唸完"
    if dur > 25:
        wav = wav[: 25 * sr]
    if float(np.mean(np.abs(wav) > 0.99)) > 0.01:
        return None, "聲音爆掉了，離麥克風遠一點再錄一次"
    wav = wav / (np.abs(wav).max() or 1) * 0.9
    return wav, None


async def create(request):
    reader = await request.multipart()
    fields, raw = {}, b""
    async for part in reader:
        if part.name == "audio":
            while True:
                chunk = await part.read_chunk(65536)
                if not chunk:
                    break
                raw += chunk
                if len(raw) > MAX_AUDIO:
                    return web.json_response({"ok": False, "error": "錄音太長了"}, status=413)
        else:
            fields[part.name] = (await part.text())[:200]
    if fields.get("consent_self") != "1" or fields.get("consent_use") != "1":
        return web.json_response({"ok": False, "error": "要勾選兩個同意才能建立"}, status=400)
    name = re.sub(r"[\x00-\x1f<>【】]", "", fields.get("name", "")).strip()[:12] or "我的聲音"
    gender = fields.get("gender") if fields.get("gender") in ("female", "male") else "female"
    if not raw:
        return web.json_response({"ok": False, "error": "沒有收到錄音"}, status=400)
    if os.path.isdir(VOICES_DIR) and sum(1 for x in os.listdir(VOICES_DIR) if x.startswith("u_")) >= MAX_USER_VOICES:
        return web.json_response({"ok": False, "error": "自己錄的聲音太多了，請先刪掉一些"}, status=429)
    loop = asyncio.get_running_loop()
    try:
        wav, sr = await loop.run_in_executor(None, _decode, raw)
    except Exception as e:
        logger.warning(f"[gf-voice] 錄音解不開：{e}")
        return web.json_response({"ok": False, "error": "這個錄音格式讀不了，換個瀏覽器（建議 Chrome）再試"}, status=400)
    wav, err = _clean(wav, sr)
    if err:
        return web.json_response({"ok": False, "error": err}, status=400)
    vid, key = "u_" + secrets.token_hex(8), secrets.token_urlsafe(18)
    d = os.path.join(VOICES_DIR, vid)
    os.makedirs(d, exist_ok=True)
    import soundfile as sf
    sf.write(os.path.join(d, "neutral.wav"), wav, sr, subtype="PCM_16")
    with open(os.path.join(d, "neutral.txt"), "w", encoding="utf-8") as f:
        f.write(RECORD_TEXT)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "gender": gender, "owner": _hash(key), "created": int(time.time()),
                   "consent": {"self": True, "use": True, "live_recording": True}}, f, ensure_ascii=False)
    for p in os.listdir(d):
        os.chmod(os.path.join(d, p), 0o644)
    os.chmod(d, 0o755)
    logger.info(f"[gf-voice] 新的自己的聲音 {vid}（{len(wav) / sr:.1f} 秒）")
    return web.json_response({"ok": True, "id": vid, "key": key, "name": name, "gender": gender})


async def delete(request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    vid, key = str(body.get("id", "")), str(body.get("key", ""))
    if not USER_RE.match(vid):
        return web.json_response({"ok": False, "error": "只能刪自己錄的聲音"}, status=400)
    d = os.path.join(VOICES_DIR, vid)
    try:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            owner = json.load(f).get("owner")
    except Exception:
        return web.json_response({"ok": True, "status": "missing"})
    if not key or _hash(key) != owner:
        return web.json_response({"ok": False, "error": "金鑰不對"}, status=403)
    shutil.rmtree(d, ignore_errors=True)
    for k in [k for k in _preview_cache if k[0] == vid]:
        _preview_cache.pop(k, None)
    logger.info(f"[gf-voice] 已刪除 {vid}")
    return web.json_response({"ok": True})


def valid_voice(v) -> str:
    """/offer 帶來的 voice：內建或自己錄的、而且存在才用，否則 female"""
    v = str(v or "")
    if v == "iai" and iai_on():
        return v
    if ID_RE.match(v) and os.path.isfile(os.path.join(VOICES_DIR, v, "neutral.wav")):
        return v
    return "female"


def setup(app):
    os.makedirs(VOICES_DIR, exist_ok=True)
    app.router.add_get("/tts/voices", voices)
    app.router.add_get("/tts/preview", preview)
    app.router.add_post("/voice/create", create)
    app.router.add_post("/voice/delete", delete)
    logger.info(f"[gf-voice] 聲音資料夾 {VOICES_DIR}")
