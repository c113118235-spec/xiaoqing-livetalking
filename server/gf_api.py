"""給外部網站串接的 API：在別人的網頁裡放會說話的小晴（有臉、有嘴型）。

流程（細節見 docs/api_v1.md）：
  1. 對方的「後端」用 API 金鑰（sk_...）換一張短效入場券：
       POST /api/v1/tokens            Authorization: Bearer sk_...   → {token, expires_in}
  2. 對方的「網頁」把入場券交給 xiaoqing-embed.js（或自己做 WebRTC）：
       POST /api/v1/sessions          Authorization: Bearer <token>  {sdp, type} → {session_id, session_token, sdp, type}
  3. 之後用 session_token（網頁）或 sk_（後端）控制：
       POST   /api/v1/sessions/{id}/say        {text, mode: "echo"|"chat", interrupt}
       POST   /api/v1/sessions/{id}/interrupt
       GET    /api/v1/sessions/{id}            → {speaking, expires_at}
       DELETE /api/v1/sessions/{id}
  另外：GET /api/v1/ice（WebRTC 用的 STUN/TURN）、GET /api/v1/voices、GET /embed（嵌入用的頁面）

安全：
  - 金鑰存在 data/api/keys.json，只存 sha256；用 tools/api_keys.py 建立／停用，原文只在建立時印一次
  - sk_ 只能在後端用：帶 Origin（= 從瀏覽器送）的 sk_ 請求一律拒絕
  - 入場券、session_token 用伺服器自己的密鑰（data/api/secret，第一次自動產生）簽名，有到期時間；
    入場券只能開一次連線；session_token 只能控制自己那一條連線
  - 每把金鑰有：允許的網域（origins）、同時幾條連線、每分鐘幾次、每條連線最長幾分鐘、能不能用 chat 模式
  - 外部連線只能用預設人物和內建聲音（不能用使用者自己錄的聲音、自訂照片學伴）
  - 記錄只寫金鑰名稱、動作、成功或失敗，不寫說話內容和金鑰
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections import defaultdict, deque

from aiohttp import web

from utils.logger import logger

API_DIR = os.getenv("GF_API_DIR", os.path.join("data", "api"))
KEYS_FILE = os.path.join(API_DIR, "keys.json")
SECRET_FILE = os.path.join(API_DIR, "secret")
TOKEN_TTL = int(os.getenv("GF_API_TOKEN_TTL", "120"))          # 入場券幾秒內要拿去開連線
MAX_TEXT = int(os.getenv("GF_API_MAX_TEXT", "300"))             # 一次最多幾個字
EMBED_HTML = os.path.join("web", "embed.html")
KEY_RE = re.compile(r"^sk_([a-z0-9][a-z0-9-]{1,31})_([A-Za-z0-9_-]{20,})$")
DEFAULTS = {"enabled": True, "origins": [], "max_sessions": 1, "rpm": 60, "minutes": 30, "chat": True}

_keys = {"mtime": None, "data": {}}
_secret = {"v": None}
_used_nonces: dict = {}                 # 入場券的 nonce → 到期時間（只能用一次）
_hits = defaultdict(deque)              # 金鑰名稱 → 最近 60 秒的請求時間
_sessions: dict = {}                    # session_id → {key, pc, expires, nonce}


# ───────────── 金鑰、簽名 ─────────────
def _load_keys() -> dict:
    try:
        m = os.path.getmtime(KEYS_FILE)
    except OSError:
        _keys.update(mtime=None, data={})
        return {}
    if m != _keys["mtime"]:
        try:
            with open(KEYS_FILE, encoding="utf-8") as f:
                raw = json.load(f).get("keys", {})
            _keys.update(mtime=m, data={k: {**DEFAULTS, **v} for k, v in raw.items()})
            logger.info(f"[api] 讀到 {len(raw)} 把外部金鑰")
        except Exception as e:
            logger.warning(f"[api] 讀不懂 {KEYS_FILE}：{e}")
            _keys.update(mtime=m, data={})
    return _keys["data"]


def _server_secret() -> bytes:
    if _secret["v"]:
        return _secret["v"]
    os.makedirs(API_DIR, exist_ok=True)
    if not os.path.isfile(SECRET_FILE):
        fd = os.open(SECRET_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    with open(SECRET_FILE) as f:
        _secret["v"] = bytes.fromhex(f.read().strip())
    return _secret["v"]


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload: dict) -> str:
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(_server_secret(), body.encode(), hashlib.sha256).digest())
    return f"xq1.{body}.{sig}"


def _verify(token: str) -> dict | None:
    try:
        tag, body, sig = token.split(".")
        if tag != "xq1":
            return None
        good = _b64(hmac.new(_server_secret(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        p = json.loads(_unb64(body))
        if p.get("exp", 0) < time.time():
            return None
        return p
    except Exception:
        return None


def _check_secret_key(raw: str):
    """sk_<名稱>_<隨機>：回傳 (名稱, 設定) 或 (None, None)"""
    m = KEY_RE.match(raw or "")
    if not m:
        return None, None
    name = m.group(1)
    cfg = _load_keys().get(name)
    if not cfg or not cfg.get("enabled"):
        return None, None
    if not hmac.compare_digest(hashlib.sha256(raw.encode()).hexdigest(), str(cfg.get("sha256", ""))):
        return None, None
    return name, cfg


# ───────────── 共用檢查 ─────────────
def _err(status: int, msg: str):
    return web.json_response({"ok": False, "error": msg}, status=status)


def _bearer(request) -> str:
    h = request.headers.get("Authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else ""


def _origin_ok(request, cfg) -> bool:
    origin = request.headers.get("Origin")
    if not origin:
        return True                        # 後端呼叫沒有 Origin
    from urllib.parse import urlparse
    host = (request.headers.get("X-Forwarded-Host") or request.host or "").split(",")[0].strip().lower()
    if urlparse(origin).netloc.lower() == host:
        return True                        # 我們自己的 /embed 頁（iframe 或 App 的 WebView 裡）送的
    allowed = [o.rstrip("/") for o in cfg.get("origins") or []]
    return origin.rstrip("/") in allowed


def _rate_ok(name: str, cfg) -> bool:
    now = time.time()
    q = _hits[name]
    while q and q[0] < now - 60:
        q.popleft()
    if len(q) >= int(cfg.get("rpm", 60)):
        return False
    q.append(now)
    return True


def _log(name, action, ok, extra=""):
    logger.info(f"[api] {name} {action} {'成功' if ok else '失敗'} {extra}".rstrip())


def _auth(request, session_id: str | None = None, count: bool = True, raw: str | None = None):
    """sk_（後端）或 session_token（網頁）。回傳 (金鑰名稱, 設定, 錯誤回應)"""
    raw = raw if raw is not None else _bearer(request)
    if not raw:
        return None, None, _err(401, "缺少 Authorization: Bearer ...")
    if raw.startswith("sk_"):
        if request.headers.get("Origin"):
            return None, None, _err(403, "sk_ 金鑰只能在後端使用，不能放在網頁裡")
        name, cfg = _check_secret_key(raw)
        if not name:
            return None, None, _err(401, "金鑰不對或已停用")
    else:
        p = _verify(raw)
        if not p or p.get("t") != "s":
            return None, None, _err(401, "session_token 不對或過期")
        name = p.get("k")
        cfg = _load_keys().get(name)
        if not cfg or not cfg.get("enabled"):
            return None, None, _err(401, "金鑰已停用")
        if session_id is not None and p.get("sid") != session_id:
            return None, None, _err(403, "這張 session_token 不是這條連線的")
    if not _origin_ok(request, cfg):
        return None, None, _err(403, "這個網域沒有被允許")
    if count and not _rate_ok(name, cfg):
        return None, None, _err(429, "太頻繁了，請稍後再試")
    if session_id is not None:
        s = _sessions.get(session_id)
        if not s or s["key"] != name:
            return None, None, _err(404, "找不到這條連線")
    return name, cfg, None


def _live_count(name: str) -> int:
    from server.session_manager import session_manager
    for sid in [k for k, v in _sessions.items() if not session_manager.has_session(k) and v["pc"].connectionState in ("closed", "failed")]:
        _sessions.pop(sid, None)
    return sum(1 for v in _sessions.values() if v["key"] == name)


def _builtin_voice(v, cfg) -> str:
    from server import gf_voice
    v = str(v or "female")
    if v.startswith("u_"):                              # 使用者自己錄的聲音不給外部用
        return "female"
    if v.startswith("iai") and not cfg.get("iai", False):
        return "female"
    return gf_voice.valid_voice(v)


# ───────────── 路由 ─────────────
async def create_token(request):
    """後端：POST /api/v1/tokens {voice?, persona?, greet?} → 一張只能開一次連線的入場券"""
    raw = _bearer(request)
    if request.headers.get("Origin"):
        return _err(403, "sk_ 金鑰只能在後端使用，不能放在網頁裡")
    name, cfg = _check_secret_key(raw)
    if not name:
        return _err(401, "金鑰不對或已停用")
    if not _rate_ok(name, cfg):
        return _err(429, "太頻繁了，請稍後再試")
    try:
        body = await request.json() if request.can_read_body else {}
    except Exception:
        body = {}
    payload = {"t": "e", "k": name, "n": secrets.token_hex(8), "exp": int(time.time()) + TOKEN_TTL,
               "v": _builtin_voice(body.get("voice"), cfg),
               "p": str(body.get("persona") or "")[:40], "g": bool(body.get("greet", False))}
    _log(name, "tokens", True)
    return web.json_response({"ok": True, "token": _sign(payload), "expires_in": TOKEN_TTL, "voice": payload["v"]})


async def _open(request, read_offer):
    """入場券檢查 → 建立小晴連線 → 回傳 (資訊 dict, None) 或 (None, 錯誤回應)"""
    p = _verify(_bearer(request))
    if not p or p.get("t") != "e":
        return None, _err(401, "入場券不對或過期，請重新向你的後端要一張")
    name = p["k"]
    cfg = _load_keys().get(name)
    if not cfg or not cfg.get("enabled"):
        return None, _err(401, "金鑰已停用")
    if not _origin_ok(request, cfg):
        return None, _err(403, "這個網域沒有被允許")
    if not _rate_ok(name, cfg):
        return None, _err(429, "太頻繁了，請稍後再試")
    now = time.time()
    for n in [n for n, e in _used_nonces.items() if e < now]:
        _used_nonces.pop(n, None)
    if p["n"] in _used_nonces:
        return None, _err(409, "這張入場券已經用過了")
    if _live_count(name) >= int(cfg.get("max_sessions", 1)):
        return None, _err(429, "同時連線數已滿")
    try:
        sdp, typ = await read_offer(request)
        if "m=video" not in sdp and "m=audio" not in sdp:
            raise ValueError
    except Exception:
        return None, _err(400, "需要 WebRTC offer（SDP）")
    _used_nonces[p["n"]] = p["exp"]

    from aiortc import RTCSessionDescription
    from server.session_manager import session_manager, MaxSessionError
    rtc = request.app["rtc_manager"]
    try:
        sid = await session_manager.create_session({"voice": p.get("v", "female")})
    except MaxSessionError:
        _log(name, "sessions", False, "伺服器滿了")
        return None, _err(503, "小晴現在太忙了（伺服器連線數已滿），請稍後再試")
    s = session_manager.get_session(sid)
    if p.get("p"):
        try:
            import llm
            ok, _label, greeting = llm.set_persona(s, p["p"])
            if ok and p.get("g") and greeting:
                s.put_msg_txt(greeting, {})
        except Exception:
            logger.exception("[api] 設定人設失敗")
    pc = await rtc._create_pc_and_answer(s, sid, RTCSessionDescription(sdp=sdp, type=typ))
    minutes = float(cfg.get("minutes", 30))
    expires = int(now + minutes * 60)
    _sessions[sid] = {"key": name, "pc": pc, "expires": expires}
    asyncio.get_event_loop().create_task(_expire(sid, minutes * 60))
    _log(name, "sessions", True, f"（{_live_count(name)} 條）")
    return {"session_id": sid, "session_token": _sign({"t": "s", "k": name, "sid": sid, "exp": expires}),
            "expires_at": expires, "origins": cfg.get("origins") or [], "chat": bool(cfg.get("chat", True)),
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, None


async def _json_offer(request):
    body = await request.json()
    return str(body["sdp"]), str(body.get("type", "offer"))


async def _raw_offer(request):
    return await request.text(), "offer"


async def create_session(request):
    """POST /api/v1/sessions {sdp, type}（Authorization: Bearer <入場券>）→ JSON（含 answer SDP）"""
    info, err = await _open(request, _json_offer)
    if err:
        return err
    return web.json_response({"ok": True, **info})


async def whep(request):
    """標準 WHEP：POST /api/v1/whep（Content-Type: application/sdp，Authorization: Bearer <入場券>）
       → 201，內容是 answer SDP；Location、X-Session-ID、X-Session-Token 在回應標頭"""
    info, err = await _open(request, _raw_offer)
    if err:
        return err
    sid = info["session_id"]
    return web.Response(status=201, body=info["sdp"].encode(), headers={"Content-Type": "application/sdp",
        "Location": f"/api/v1/whep/{sid}", "X-Session-ID": sid, "X-Session-Token": info["session_token"],
        "X-Session-Expires": str(info["expires_at"])})


async def whep_delete(request):
    """WHEP 結束：DELETE /api/v1/whep/{id}（Bearer session_token 或 sk_）"""
    sid = request.match_info["sid"]
    name, cfg, err = _auth(request, sid)
    if err:
        return err
    await _close(sid)
    return web.Response(status=200)


async def _close(sid: str):
    s = _sessions.pop(sid, None)
    if not s:
        return
    try:
        await s["pc"].close()
    except Exception:
        pass
    from server.session_manager import session_manager
    session_manager.remove_session(sid)


async def _expire(sid: str, seconds: float):
    await asyncio.sleep(seconds)
    if sid in _sessions:
        logger.info(f"[api] {_sessions[sid]['key']} 連線到時間了，自動結束")
        await _close(sid)


def _avatar(sid):
    from server.session_manager import session_manager
    return session_manager.get_session(sid)


EMOTIONS = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry", "soft": "soft",
            "平常": "neutral", "開心": "happy", "難過": "sad", "生氣": "angry", "溫柔": "soft"}


def _action_names() -> list:
    try:
        from server import gf_pose
        return list(gf_pose.actions().get("actions", {}))
    except Exception:
        return []


def _segments(body) -> list | str:
    """把 /say 的內容整理成 [(文字, 情緒, 動作)]；格式不對回傳錯誤訊息（字串）
      - {text, emotion?, action?}
      - {segments: [{text, emotion?, action?}, ...]}   一句一個情緒／動作
      - text 裡也可以直接帶 LLM 的標籤：【開心】【揮手】…（會拿掉不念）
    """
    import llm
    raw = body.get("segments")
    if raw is None:
        raw = [{"text": body.get("text", ""), "emotion": body.get("emotion"), "action": body.get("action")}]
    if not isinstance(raw, list) or not raw or len(raw) > 20:
        return "segments 要是 1～20 段的清單"
    acts = set(_action_names())
    out, total = [], 0
    for seg in raw:
        if not isinstance(seg, dict):
            return "segments 的每一段要是 {text, emotion, action}"
        text = str(seg.get("text") or "")
        emo = seg.get("emotion")
        if emo is not None and str(emo) not in EMOTIONS:
            return "emotion 只能是 neutral、happy、sad、angry、soft（或 平常、開心、難過、生氣、溫柔）"
        act = seg.get("action")
        if act is not None and str(act) not in acts:
            return f"沒有這個動作：{act}（可用：{', '.join(sorted(acts)) or '無'}）"
        tags = llm.EMO_TAG.findall(text)                       # 文字裡的【開心】
        if emo is None and tags:
            emo = llm.EMO_MAP[tags[-1]]
        tag_acts = [llm.ACT_MAP[t] for t in llm.ACT_TAG.findall(text) if llm.ACT_MAP[t] in acts]
        if act is None and tag_acts:
            act = tag_acts[0]
        clean = llm._clean(text)
        total += len(clean)
        if not clean and not act:
            continue
        out.append((clean, EMOTIONS.get(str(emo), "neutral") if emo is not None else "neutral", act))
    if not out:
        return "沒有要念的文字或要做的動作"
    if total > MAX_TEXT:
        return f"一次最多 {MAX_TEXT} 個字"
    return out


def _speak(s, segs):
    """依序：先做動作（等做完，最多 8 秒），再用那個情緒念這段"""
    ctl = getattr(s, "_gf_pose", None)
    for text, emo, act in segs:
        if act and ctl is not None:
            ok, why = ctl.request(act)
            if not ok:
                logger.info(f"[api] 動作 {act}：{why}")
            t0 = time.time()
            while ctl.busy() and time.time() - t0 < 8:
                time.sleep(0.05)
        if text:
            s.put_msg_txt(text, {"tts": {"emotion": emo}})


async def say(request):
    """POST /api/v1/sessions/{id}/say
       {text, emotion?, action?, interrupt?}  或  {segments: [{text, emotion, action}], interrupt?}
       mode: "echo"（預設，照念：你的 LLM 回答好的文字）| "chat"（讓小晴自己回答）"""
    sid = request.match_info["sid"]
    name, cfg, err = _auth(request, sid)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return _err(400, "需要 JSON")
    mode = body.get("mode", "echo")
    if mode not in ("echo", "chat"):
        return _err(400, "mode 只能是 echo（照念）或 chat（讓小晴回答）")
    if mode == "chat" and not cfg.get("chat", True):
        return _err(403, "這把金鑰不能用 chat 模式")
    s = _avatar(sid)
    if s is None:
        return _err(404, "找不到這條連線")
    if mode == "chat":
        text = str(body.get("text", "")).strip()
        if not text or len(text) > MAX_TEXT:
            return _err(400, f"text 要有 1～{MAX_TEXT} 個字")
        if body.get("interrupt"):
            s.flush_talk()
        asyncio.get_event_loop().run_in_executor(None, request.app.get("llm_response"), text, s, {})
        _log(name, "say/chat", True, f"{len(text)} 字")
        return web.json_response({"ok": True})
    segs = _segments(body)
    if isinstance(segs, str):
        return _err(400, segs)
    if body.get("interrupt"):
        s.flush_talk()
    asyncio.get_event_loop().run_in_executor(None, _speak, s, segs)
    _log(name, "say/echo", True, f"{len(segs)} 段 {sum(len(t) for t, _, _ in segs)} 字")
    return web.json_response({"ok": True, "segments": len(segs)})


async def actions(request):
    """GET /api/v1/actions：可以用的動作和情緒"""
    return web.json_response({"ok": True, "actions": sorted(_action_names()),
                              "emotions": ["neutral", "happy", "sad", "angry", "soft"]})


async def interrupt(request):
    sid = request.match_info["sid"]
    name, cfg, err = _auth(request, sid)
    if err:
        return err
    s = _avatar(sid)
    if s is not None:
        s.flush_talk()
    return web.json_response({"ok": True})


async def status(request):
    sid = request.match_info["sid"]
    name, cfg, err = _auth(request, sid, count=False)      # 網頁會一直問「在說話嗎」，不算進每分鐘次數
    if err:
        return err
    s = _avatar(sid)
    return web.json_response({"ok": True, "session_id": sid, "alive": s is not None,
                              "speaking": bool(s.is_speaking()) if s is not None else False,
                              "expires_at": _sessions.get(sid, {}).get("expires")})


async def end(request):
    sid = request.match_info["sid"]
    name, cfg, err = _auth(request, sid)
    if err:
        return err
    await _close(sid)
    _log(name, "end", True)
    return web.json_response({"ok": True})


async def ice(request):
    """WebRTC 用的 STUN/TURN；要有效的入場券或 session_token，免得 TURN 帳號被拿去亂用"""
    p = _verify(_bearer(request))
    if not p or p.get("t") not in ("e", "s"):
        return _err(401, "需要入場券")
    from server.gf_routes import get_ice_servers
    return web.json_response({"iceServers": await get_ice_servers()})


async def voices(request):
    from server import gf_voice
    out = []
    for d in sorted(os.listdir(gf_voice.VOICES_DIR)) if os.path.isdir(gf_voice.VOICES_DIR) else []:
        if d.startswith("u_") or not os.path.isfile(os.path.join(gf_voice.VOICES_DIR, d, "neutral.wav")):
            continue
        try:
            with open(os.path.join(gf_voice.VOICES_DIR, d, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        out.append({"id": d, "name": meta.get("name", d), "gender": meta.get("gender", "")})
    return web.json_response({"ok": True, "voices": out})


async def embed_page(request):
    """嵌入用的頁面；frame-ancestors 只允許金鑰設定裡的網域把它放進 iframe"""
    origins = sorted({o.rstrip("/") for c in _load_keys().values() if c.get("enabled") for o in c.get("origins") or []})
    with open(EMBED_HTML, encoding="utf-8") as f:
        html = f.read()
    return web.Response(text=html, content_type="text/html", headers={
        "Content-Security-Policy": "frame-ancestors " + (" ".join(origins) if origins else "'none'"),
        "Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


# ───────────── 對方送「已經生好的語音」，小晴只對嘴 ─────────────
AUDIO_SR = 16000                                           # 小晴內部算嘴型用 16 kHz
CHUNK = 320                                                # 20 ms
MAX_AUDIO_SEC = float(os.getenv("GF_API_MAX_AUDIO_SEC", "60"))       # 一段最長幾秒
MAX_QUEUE_SEC = float(os.getenv("GF_API_MAX_QUEUE_SEC", "90"))       # 排隊還沒播的最多幾秒
MAX_AUDIO_BYTES = int(os.getenv("GF_API_MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))
# 串流時先存這麼多才開始送進對嘴流程：網路一抖、某一片晚到，排隊裡還有存貨，就不會插進靜音造成「卡一下」
PREBUFFER_SEC = float(os.getenv("GF_API_PREBUFFER_MS", "300")) / 1000


def _queued_sec(s) -> float:
    try:
        return s.asr.queue.qsize() * CHUNK / AUDIO_SR
    except Exception:
        return 0.0


def _to_float(data: bytes, fmt: str, channels: int):
    import numpy as np
    if fmt == "pcm_f32le":
        a = np.frombuffer(data[: len(data) // 4 * 4], dtype="<f4").astype(np.float32)
    else:                                                    # pcm_s16le
        a = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        a = a[: len(a) // channels * channels].reshape(-1, channels).mean(axis=1)
    return a


def _resample(a, sr: int):
    if sr == AUDIO_SR or len(a) == 0:
        return a
    import resampy
    return resampy.resample(a, sr, AUDIO_SR).astype("float32")


class _Feeder:
    """把 16 kHz 音訊切成 20 ms 一片送進小晴的對嘴流程；第一片標 start、最後一片標 end"""

    def __init__(self, s):
        import numpy as np
        self.s, self.np = s, np
        self.buf = np.zeros(0, np.float32)
        self.first, self.total = True, 0

    def push(self, a, last=False):
        np = self.np
        self.buf = np.concatenate([self.buf, a]) if len(a) else self.buf
        if self.first and not last and len(self.buf) < PREBUFFER_SEC * AUDIO_SR:
            return                                       # 還在預先緩衝
        if last and len(self.buf) % CHUNK:
            self.buf = np.concatenate([self.buf, np.zeros(CHUNK - len(self.buf) % CHUNK, np.float32)])
        n = len(self.buf) // CHUNK
        if last and n == 0 and not self.first:          # 結尾剛好沒剩：補一片靜音帶 end
            self.buf, n = np.zeros(CHUNK, np.float32), 1
        for i in range(n):
            ev = {}
            if self.first:
                ev, self.first = {"status": "start"}, False
            elif last and i == n - 1:
                ev = {"status": "end"}
            self.s.put_audio_frame(self.buf[i * CHUNK:(i + 1) * CHUNK], ev)
        self.total += n * CHUNK
        self.buf = self.buf[n * CHUNK:]

    @property
    def seconds(self):
        return self.total / AUDIO_SR


async def _do_action(s, act, wait=True):
    ctl = getattr(s, "_gf_pose", None)
    if not act or ctl is None:
        return
    ok, why = ctl.request(act)
    if not ok:
        logger.info(f"[api] 動作 {act}：{why}")
        return
    t0 = time.time()
    while wait and ctl.busy() and time.time() - t0 < 8:
        await asyncio.sleep(0.05)


async def audio_upload(request):
    """POST /api/v1/sessions/{id}/audio：整段語音（wav／mp3／ogg…，或 ?format=pcm_s16le&sample_rate=16000 的 raw PCM）
       ?action=wave&interrupt=1"""
    sid = request.match_info["sid"]
    name, cfg, err = _auth(request, sid)
    if err:
        return err
    s = _avatar(sid)
    if s is None:
        return _err(404, "找不到這條連線")
    if request.content_length and request.content_length > MAX_AUDIO_BYTES:
        return _err(413, f"音檔最大 {MAX_AUDIO_BYTES // 1024 // 1024} MB")
    data = await request.read()
    if not data or len(data) > MAX_AUDIO_BYTES:
        return _err(413 if data else 400, "音檔是空的或太大")
    q = request.query
    act = q.get("action")
    if act and act not in _action_names():
        return _err(400, f"沒有這個動作：{act}")
    fmt = q.get("format", "")
    try:
        if fmt in ("pcm_s16le", "pcm_f32le"):
            sr = int(q.get("sample_rate", AUDIO_SR))
            a = _to_float(data, fmt, int(q.get("channels", 1)))
        else:
            from tts.gftts import _decode                         # soundfile → 失敗再用 PyAV（mp3 等）
            a, sr = _decode(data)
            a = a.astype("float32")
        if not 8000 <= sr <= 48000:
            raise ValueError("sample_rate 要在 8000～48000")
        a = await asyncio.get_event_loop().run_in_executor(None, _resample, a, sr)
    except Exception as e:
        return _err(400, f"讀不懂這個音檔：{e}")
    sec = len(a) / AUDIO_SR
    if sec > MAX_AUDIO_SEC:
        return _err(413, f"一段最長 {MAX_AUDIO_SEC:.0f} 秒")
    if q.get("interrupt") in ("1", "true"):
        s.flush_talk()
    if _queued_sec(s) + sec > MAX_QUEUE_SEC:
        return _err(429, "排隊的語音太多了，等她說完或先 interrupt")
    await _do_action(s, act)
    f = _Feeder(s)
    f.push(a, last=True)
    _log(name, "audio", True, f"{sec:.1f} 秒")
    return web.json_response({"ok": True, "seconds": round(sec, 2)})


async def audio_stream(request):
    """WebSocket /api/v1/sessions/{id}/audio/stream
       認證：後端用 Authorization: Bearer sk_...；網頁（瀏覽器的 WebSocket 不能帶標頭）用 ?token=<session_token>
       文字訊息（JSON）：
         {"type":"start", "format":"pcm_s16le"|"pcm_f32le", "sample_rate":16000, "channels":1, "action":"wave"?, "interrupt":true?}
         {"type":"end"}                一段結束（最後不足 20 ms 的部分補靜音送出）
         {"type":"interrupt"}          停掉正在說的（含排隊中的）
         {"type":"action","action":"clap"}
       二進位訊息：start 之後的 PCM 音訊（任意長度，建議 20～200 ms 一片）
       回覆：{"type":"started"} / {"type":"done","seconds":1.23} / {"type":"error","error":"…"}"""
    sid = request.match_info["sid"]
    qtok = request.query.get("token")
    if qtok and qtok.startswith("sk_"):
        return _err(403, "sk_ 金鑰不能放在網址裡；後端請用 Authorization 標頭")
    name, cfg, err = _auth(request, sid, raw=_bearer(request) or qtok or "")
    if err:
        return err
    s = _avatar(sid)
    if s is None:
        return _err(404, "找不到這條連線")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=512 * 1024)
    await ws.prepare(request)
    _log(name, "audio/stream", True, "連上")
    acts = set(_action_names())
    st = None                                          # 目前這一段的設定
    seg_sec = 0.0

    async def send(obj):
        if not ws.closed:
            await ws.send_json(obj)

    async for msg in ws:
        if sid not in _sessions or _avatar(sid) is None:
            await send({"type": "error", "error": "連線已結束"})
            break
        if msg.type == web.WSMsgType.TEXT:
            try:
                m = json.loads(msg.data)
                typ = m.get("type")
            except Exception:
                await send({"type": "error", "error": "文字訊息要是 JSON"})
                continue
            if typ == "start":
                fmt = m.get("format", "pcm_s16le")
                sr, ch = int(m.get("sample_rate", AUDIO_SR)), int(m.get("channels", 1))
                if fmt not in ("pcm_s16le", "pcm_f32le") or not 8000 <= sr <= 48000 or ch not in (1, 2):
                    await send({"type": "error", "error": "format 要是 pcm_s16le 或 pcm_f32le，sample_rate 8000～48000，channels 1 或 2"})
                    continue
                act = m.get("action")
                if act and act not in acts:
                    await send({"type": "error", "error": f"沒有這個動作：{act}"})
                    continue
                if m.get("interrupt"):
                    s.flush_talk()
                await _do_action(s, act)
                st = {"fmt": fmt, "sr": sr, "ch": ch, "feeder": _Feeder(s), "pend": b""}
                seg_sec = 0.0
                await send({"type": "started"})
            elif typ == "end":
                if st:
                    a = _to_float(st["pend"], st["fmt"], st["ch"]) if st["pend"] else _to_float(b"", "pcm_s16le", 1)
                    st["feeder"].push(_resample(a, st["sr"]), last=True)
                    await send({"type": "done", "seconds": round(st["feeder"].seconds, 2)})
                    _log(name, "audio/stream", True, f"{st['feeder'].seconds:.1f} 秒")
                st = None
            elif typ == "interrupt":
                s.flush_talk()
                st = None
                await send({"type": "interrupted"})
            elif typ == "action":
                if m.get("action") not in acts:
                    await send({"type": "error", "error": f"沒有這個動作：{m.get('action')}"})
                else:
                    await _do_action(s, m["action"], wait=False)
            else:
                await send({"type": "error", "error": f"不認得的 type：{typ}"})
        elif msg.type == web.WSMsgType.BINARY:
            if not st:
                await send({"type": "error", "error": "先送 {\"type\":\"start\"} 再送音訊"})
                continue
            width = (4 if st["fmt"] == "pcm_f32le" else 2) * st["ch"]
            data = st["pend"] + msg.data
            use = len(data) // width * width
            st["pend"] = data[use:]
            # 非 16 kHz 時累積到 0.1 秒再轉，減少每片轉換的接縫
            if st["sr"] != AUDIO_SR and use < st["sr"] // 10 * width:
                st["pend"] = data
                continue
            a = _to_float(data[:use], st["fmt"], st["ch"])
            seg_sec += len(a) / st["sr"]
            if seg_sec > MAX_AUDIO_SEC or _queued_sec(s) > MAX_QUEUE_SEC:
                await send({"type": "error", "error": f"一段最長 {MAX_AUDIO_SEC:.0f} 秒，或排隊太多；這段後面的音訊已丟掉"})
                st = None
                continue
            st["feeder"].push(_resample(a, st["sr"]))
        elif msg.type == web.WSMsgType.ERROR:
            break
    _log(name, "audio/stream", True, "斷開")
    return ws


def setup(app):
    app.router.add_post("/api/v1/tokens", create_token)
    app.router.add_post("/api/v1/sessions", create_session)
    app.router.add_post("/api/v1/whep", whep)
    app.router.add_delete("/api/v1/whep/{sid}", whep_delete)
    app.router.add_get("/api/v1/sessions/{sid}", status)
    app.router.add_delete("/api/v1/sessions/{sid}", end)
    app.router.add_post("/api/v1/sessions/{sid}/say", say)
    app.router.add_post("/api/v1/sessions/{sid}/interrupt", interrupt)
    app.router.add_get("/api/v1/ice", ice)
    app.router.add_get("/api/v1/voices", voices)
    app.router.add_get("/api/v1/actions", actions)
    app.router.add_post("/api/v1/sessions/{sid}/audio", audio_upload)
    app.router.add_get("/api/v1/sessions/{sid}/audio/stream", audio_stream)
    app.router.add_get("/embed", embed_page)
