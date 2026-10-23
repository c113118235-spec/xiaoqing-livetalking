"""小晴版加的路由與設定（放在 server/gf_routes.py，由 server/routes.py 呼叫 setup_gf_routes(app)）

1. POST /asr        網頁錄好的聲音 → DGX 的 asr（Whisper, 8001）→ {"text": ...}
2. GET  /iceservers 給網頁用的 STUN/TURN 清單（有設定 Cloudflare TURN 才會有 TURN）
3. 伺服器端 WebRTC 也加上 TURN，讓校外（沒有 VPN）的瀏覽器也連得到影像
4. 公開網址的門禁：從 Cloudflare Tunnel 進來的請求要先輸入通關碼（GF_ACCESS_CODE），
   而且只開放小晴網頁需要的路徑；LiveTalking 的管理、上傳 avatar 等 API 一律擋掉。
   區網 / VPN 直接連 8010 不受影響。

環境變數（放在 ~/xiaozhi-speech/.env，用 --env-file 帶進來，不要打在指令上）：
  GF_ACCESS_CODE        通關碼（沒設的話，公開網址一律拒絕）
  CF_TURN_KEY_ID        Cloudflare Realtime TURN 的 Key ID
  CF_TURN_API_TOKEN     同一把 key 的 API Token
"""
import asyncio
import hashlib
import hmac
import html
import os
import time

import aiohttp
from aiohttp import web

from utils.logger import logger

ASR_URL = os.getenv("ASR_URL", "http://127.0.0.1:8001/v1/audio/transcriptions")
ASR_KEY = os.getenv("ASR_API_KEY", "")
MAX_BYTES = 5 * 1024 * 1024

ACCESS_CODE = os.getenv("GF_ACCESS_CODE", "")
TURN_KEY_ID = os.getenv("CF_TURN_KEY_ID", "")
TURN_TOKEN = os.getenv("CF_TURN_API_TOKEN", "")
TURN_TTL = 24 * 3600
COOKIE = "gf_pass"
COOKIE_VALUE = hashlib.sha256(("xiaoqing:" + ACCESS_CODE).encode()).hexdigest() if ACCESS_CODE else ""

# 從公開網址進來時，通過門禁後只開放這些路徑
PUBLIC_PATHS = {"/", "/xiaoqing.html", "/offer", "/human", "/sse", "/asr",
                "/interrupt_talk", "/is_speaking", "/iceservers"}


# ───────────── 1. 語音辨識代轉 ─────────────
async def asr(request: web.Request):
    try:
        form = await request.post()
        f = form.get("file")
        if f is None or not hasattr(f, "file"):
            return web.json_response({"error": "沒有收到聲音檔"}, status=400)
        audio = f.file.read()
        if not audio or len(audio) > MAX_BYTES:
            return web.json_response({"error": "聲音檔是空的或太大"}, status=400)

        out = aiohttp.FormData()
        out.add_field("file", audio, filename=f.filename or "voice.webm",
                      content_type=f.content_type or "application/octet-stream")
        out.add_field("language", "zh")
        headers = {"Authorization": f"Bearer {ASR_KEY}"} if ASR_KEY else {}

        t = time.time()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
            async with s.post(ASR_URL, data=out, headers=headers) as r:
                if r.status != 200:
                    msg = (await r.text())[:200]
                    logger.warning(f"[gf-asr] asr 回 {r.status}: {msg}")
                    return web.json_response({"error": f"語音辨識失敗（{r.status}）"}, status=502)
                j = await r.json()
        text = (j.get("text") or "").strip()
        logger.info(f"[gf-asr] {len(audio) / 1024:.0f} KB, {time.time() - t:.2f}s：{text}")
        return web.json_response({"text": text})
    except Exception as e:
        logger.exception("[gf-asr] exception:")
        return web.json_response({"error": str(e)}, status=500)


# ───────────── 2./3. Cloudflare TURN ─────────────
_turn_cache = {"t": 0.0, "servers": None}


def _no_port53(servers):
    """瀏覽器會擋 port 53，留著只會讓連線多等，先拿掉"""
    out = []
    for s in servers:
        urls = s.get("urls") or []
        urls = [urls] if isinstance(urls, str) else urls
        urls = [u for u in urls if ":53?" not in u and not u.endswith(":53")]
        if urls:
            out.append({**s, "urls": urls})
    return out


async def get_ice_servers():
    """回傳 [{urls, username?, credential?}, ...]；沒設定 TURN 就只有 STUN"""
    stun = [{"urls": ["stun:stun.cloudflare.com:3478", "stun:stun.l.google.com:19302"]}]
    if not (TURN_KEY_ID and TURN_TOKEN):
        return stun
    if _turn_cache["servers"] and time.time() - _turn_cache["t"] < TURN_TTL / 2:
        return _turn_cache["servers"]
    url = f"https://rtc.live.cloudflare.com/v1/turn/keys/{TURN_KEY_ID}/credentials/generate-ice-servers"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.post(url, json={"ttl": TURN_TTL},
                              headers={"Authorization": f"Bearer {TURN_TOKEN}"}) as r:
                if r.status not in (200, 201):
                    logger.warning(f"[gf-turn] Cloudflare 回 {r.status}: {(await r.text())[:200]}")
                    return stun
                j = await r.json()
        servers = j.get("iceServers") or []
        servers = [servers] if isinstance(servers, dict) else servers
        servers = _no_port53(servers)
        _turn_cache.update(t=time.time(), servers=servers)
        logger.info(f"[gf-turn] 取得 TURN 憑證（{TURN_TTL // 3600} 小時有效）")
        return servers
    except Exception as e:
        logger.warning(f"[gf-turn] 取得 TURN 憑證失敗：{e}")
        return stun


async def iceservers(request):
    return web.json_response({"iceServers": await get_ice_servers()})


def _patch_server_turn():
    """讓 LiveTalking 伺服器端的 WebRTC 也走 TURN（aiortc 只會用第一組 STUN 和第一組 TURN）"""
    if not (TURN_KEY_ID and TURN_TOKEN):
        return
    import server.rtc_manager as rm
    orig = rm.RTCConfiguration

    def config_with_turn(iceServers=None, **kw):
        servers = list(iceServers or [])
        cached = _turn_cache["servers"] or []
        for s in cached:
            urls = s.get("urls") or []
            urls = [urls] if isinstance(urls, str) else urls
            if not s.get("username"):
                continue
            # 伺服器端優先用 UDP 3478，其次 TCP；aiortc 對 turns: 的支援較不穩
            pick = ([u for u in urls if u.startswith("turn:") and "transport=udp" in u] or
                    [u for u in urls if u.startswith("turn:")])
            if pick:
                servers.append(rm.RTCIceServer(urls=pick[0], username=s["username"], credential=s["credential"]))
                break
        return orig(iceServers=servers, **kw)

    rm.RTCConfiguration = config_with_turn
    logger.info("[gf-turn] 伺服器端 WebRTC 已加上 Cloudflare TURN")


def _patch_bitrate():
    """aiortc 預設影像碼率很低（H.264 起始 1 Mbps、最高 3 Mbps；VP8 起始 0.5 Mbps、最高 1.5 Mbps），
    整張全身畫面會被壓得很糊。用環境變數調高：
      GF_VIDEO_KBPS      起始碼率（kbps），預設 4000
      GF_VIDEO_MIN_KBPS  最低碼率，預設 2000（網路差時不會掉到比這更低）
      GF_VIDEO_MAX_KBPS  最高碼率，預設 8000
    設 GF_VIDEO_KBPS=0 則維持 aiortc 原本的設定（方便比較）。"""
    start = int(os.getenv("GF_VIDEO_KBPS", "4000"))
    if start <= 0:
        logger.info("[gf-video] 使用 aiortc 預設碼率")
        return
    lo = int(os.getenv("GF_VIDEO_MIN_KBPS", "2000"))
    hi = int(os.getenv("GF_VIDEO_MAX_KBPS", "8000"))
    from aiortc.codecs import h264, vpx
    for m in (h264, vpx):
        m.DEFAULT_BITRATE, m.MIN_BITRATE, m.MAX_BITRATE = start * 1000, lo * 1000, hi * 1000
    logger.info(f"[gf-video] 影像碼率：起始 {start} kbps，範圍 {lo}–{hi} kbps")


async def _refresh_turn_loop(app):
    async def loop():
        while True:
            await get_ice_servers()
            await asyncio.sleep(TURN_TTL / 4)
    if TURN_KEY_ID and TURN_TOKEN:
        app["gf_turn_task"] = asyncio.create_task(loop())


# ───────────── 4. 公開網址門禁 ─────────────
def _from_tunnel(request) -> bool:
    return "Cf-Connecting-Ip" in request.headers or "cf-connecting-ip" in request.headers


GATE_HTML = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>小晴</title>
<style>body{{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#0B0A10;color:#fff;
font-family:"Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif}}form{{display:flex;flex-direction:column;gap:14px;width:min(320px,86vw)}}
h1{{margin:0 0 4px;font-size:24px}}small{{color:rgba(255,255,255,.55)}}input{{padding:14px 18px;border-radius:24px;border:1px solid rgba(255,255,255,.2);
background:rgba(255,255,255,.06);color:#fff;font-size:16px}}button{{padding:14px;border:none;border-radius:24px;background:#E86AA6;color:#fff;font-size:16px;font-weight:700}}
.err{{color:#F3A6D8;font-size:14px;min-height:20px}}</style></head><body>
<form method="post" action="gate"><h1>小晴 <small>AI 角色</small></h1><small>請輸入通關碼</small>
<input name="code" type="password" autocomplete="current-password" autofocus required>
<div class="err">{err}</div><button>進入</button></form></body></html>"""

_fails = {}


async def gate(request):
    if request.method == "POST":
        ip = request.headers.get("Cf-Connecting-Ip", request.remote or "")
        n, t0 = _fails.get(ip, (0, 0.0))
        if n >= 5 and time.time() - t0 < 300:
            return web.Response(text=GATE_HTML.format(err="錯太多次了，請 5 分鐘後再試"), content_type="text/html", status=429)
        code = (await request.post()).get("code", "")
        if ACCESS_CODE and hmac.compare_digest(str(code), ACCESS_CODE):
            _fails.pop(ip, None)
            resp = web.HTTPFound("/xiaoqing.html?mode=chat")
            resp.set_cookie(COOKIE, COOKIE_VALUE, max_age=7 * 86400, httponly=True, secure=True, samesite="Lax")
            raise resp
        _fails[ip] = (n + 1, time.time())
        await asyncio.sleep(1)
        return web.Response(text=GATE_HTML.format(err="通關碼不對"), content_type="text/html", status=401)
    return web.Response(text=GATE_HTML.format(err=""), content_type="text/html")


@web.middleware
async def gate_middleware(request, handler):
    if not _from_tunnel(request):
        return await handler(request)          # 區網 / VPN：照舊
    path = request.path
    if path == "/gate":
        return await handler(request)
    if not ACCESS_CODE:
        return web.Response(text="公開網址尚未設定通關碼", status=403)
    if not hmac.compare_digest(request.cookies.get(COOKIE, ""), COOKIE_VALUE):
        if request.method == "GET" and path in ("/", "/xiaoqing.html"):
            raise web.HTTPFound("/gate")
        return web.json_response({"error": "需要通關碼"}, status=401)
    if path not in PUBLIC_PATHS:
        return web.Response(text="Not found", status=404)
    if path == "/":
        raise web.HTTPFound("/xiaoqing.html?mode=chat")
    return await handler(request)


def setup_gf_routes(app):
    app.router.add_post("/asr", asr)
    app.router.add_get("/iceservers", iceservers)
    app.router.add_get("/gate", gate)
    app.router.add_post("/gate", gate)
    app.middlewares.append(gate_middleware)
    app.on_startup.append(_refresh_turn_loop)
    _patch_server_turn()
    _patch_bitrate()
    logger.info(f"[gf-asr] /asr → {ASR_URL}")
    logger.info(f"[gf-gate] 公開網址通關碼：{'已設定' if ACCESS_CODE else '未設定（公開網址會被拒絕）'}；"
                f"TURN：{'Cloudflare' if TURN_KEY_ID and TURN_TOKEN else '未設定（校外看不到影像）'}")
