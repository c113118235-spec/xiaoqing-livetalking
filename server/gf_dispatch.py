"""一台 DGX 撐更多人：開好幾個小晴程序（worker），由主程序（8010）分配連線。

為什麼：小晴所有連線都在同一個 Python 程序裡，多人同時說話時會互相卡（CPU 其實還很空）。
分到幾個程序後，每個程序只負責幾個人，就不會塞住。

怎麼運作（使用者感覺不到，網址不變）：
  1. 新連線（/offer）：主程序自己還有空位就自己接；滿了就轉給人最少的 worker，記下「這條連線在哪個 worker」
  2. 之後這條連線的請求（說話、打斷、黑板、SSE…，看 sessionid）都轉給同一個 worker
  3. 影像和聲音（WebRTC）直接從 worker 送到瀏覽器，不經過主程序

設定 data/workers.json（改了馬上生效；沒有這個檔 = 不分配，跟以前一樣）：
  {"local_max": 5, "workers": [{"url": "http://127.0.0.1:8021", "max": 5}, {"url": "http://127.0.0.1:8022", "max": 5}]}
worker 用 tools/workers.sh 啟動。
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp
from aiohttp import web

from utils.logger import logger

CONF = os.path.join("data", "workers.json")
_conf = {"mtime": None, "data": None}
_where: dict = {}                 # sessionid → worker url
_http: aiohttp.ClientSession | None = None
# 這些路徑跟某一條連線綁在一起（用 sessionid 找），要轉給那條連線所在的 worker
SESSION_PATHS = {"/human", "/interrupt_talk", "/is_speaking", "/pose", "/persona", "/board", "/board/say",
                 "/sse", "/msgpoll", "/bye", "/set_audiotype", "/record"}
HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}


def _load():
    try:
        m = os.path.getmtime(CONF)
    except OSError:
        _conf.update(mtime=None, data=None)
        return None
    if m != _conf["mtime"]:
        try:
            with open(CONF, encoding="utf-8") as f:
                d = json.load(f)
            d["workers"] = [w for w in d.get("workers", []) if w.get("url")]
            _conf.update(mtime=m, data=d)
            logger.info(f"[gf-dispatch] 主程序最多 {d.get('local_max', 5)} 人＋{len(d['workers'])} 個 worker")
        except Exception as e:
            logger.warning(f"[gf-dispatch] 讀不懂 {CONF}：{e}")
            _conf.update(mtime=m, data=None)
    return _conf["data"]


def _session():
    global _http
    if _http is None or _http.closed:
        _http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=3, sock_read=None))
    return _http


def _local_count() -> int:
    from server.session_manager import session_manager
    return sum(1 for s in session_manager.sessions.values() if s is not None)


async def _worker_count(url) -> int | None:
    try:
        async with _session().get(url.rstrip("/") + "/api/admin/sessions",
                                  timeout=aiohttp.ClientTimeout(total=2)) as r:
            d = await r.json(content_type=None)
        return len(d.get("data", {}).get("sessions", []))
    except Exception:
        return None                                   # worker 沒開或沒回應：先跳過


async def _proxy(request, base, body: bytes):
    url = base.rstrip("/") + request.rel_url.path_qs
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    try:
        async with _session().request(request.method, url, data=body or None, headers=headers) as r:
            ctype = r.headers.get("Content-Type", "")
            if "text/event-stream" in ctype:            # SSE：一路轉送
                resp = web.StreamResponse(status=r.status, headers={"Content-Type": ctype, "Cache-Control": "no-cache"})
                await resp.prepare(request)
                async for chunk in r.content.iter_any():
                    await resp.write(chunk)
                return resp
            data = await r.read()
            return web.Response(status=r.status, body=data,
                                headers={k: v for k, v in r.headers.items() if k.lower() not in HOP | {"content-encoding"}})
    except Exception as e:
        logger.warning(f"[gf-dispatch] 轉給 {base} 失敗：{e}")
        return web.json_response({"code": -1, "msg": "這條連線所在的程序沒有回應，請重新整理"}, status=502)


async def _sid_of(request) -> tuple[str, bytes]:
    sid = request.query.get("sessionid", "")
    body = b""
    if request.method == "POST" and request.can_read_body:
        body = await request.read()                    # aiohttp 會快取，之後本機的 handler 還讀得到
        if not sid and "multipart" not in (request.content_type or ""):
            try:
                sid = str(json.loads(body or b"{}").get("sessionid", ""))
            except Exception:
                sid = ""
    return sid, body


async def _offer(request, conf):
    """新連線：主程序有空位就自己接，滿了轉給人最少的 worker"""
    local_max = int(conf.get("local_max", 5))
    if _local_count() < local_max:
        return None                                    # 交給原本的 /offer
    body = await request.read()
    best, best_n = None, None
    counts = await asyncio.gather(*[_worker_count(w["url"]) for w in conf["workers"]])
    for w, n in zip(conf["workers"], counts):
        if n is None or n >= int(w.get("max", 5)):
            continue
        if best_n is None or n < best_n:
            best, best_n = w["url"], n
    if best is None:
        logger.info("[gf-dispatch] 主程序和所有 worker 都滿了")
        return web.json_response({"code": -1, "msg": "Maximum session limit reached"})
    resp = await _proxy(request, best, body)
    try:
        sid = json.loads(resp.body).get("sessionid")
        if sid:
            _where[sid] = best
            logger.info(f"[gf-dispatch] 新連線 {sid} → {best}（那邊原本 {best_n} 人）")
    except Exception:
        pass
    return resp


@web.middleware
async def dispatch_middleware(request, handler):
    conf = _load()
    if not conf or not conf.get("workers"):
        return await handler(request)
    path = request.path
    if path == "/offer" and request.method == "POST":
        r = await _offer(request, conf)
        return r if r is not None else await handler(request)
    if path in SESSION_PATHS:
        sid, body = await _sid_of(request)
        base = _where.get(sid)
        if base:
            resp = await _proxy(request, base, body)
            if path == "/bye":
                _where.pop(sid, None)
            return resp
    if path == "/api/admin/sessions" and request.method == "GET" and request.query.get("all") == "1":
        counts = await asyncio.gather(*[_worker_count(w["url"]) for w in conf["workers"]])
        return web.json_response({"local": _local_count(), "workers": {w["url"]: n for w, n in zip(conf["workers"], counts)}})
    return await handler(request)


async def _purge_loop(app):
    """每 30 秒清掉 worker 那邊已經不存在的連線記錄"""
    async def loop():
        while True:
            await asyncio.sleep(30)
            conf = _load()
            if not conf or not _where:
                continue
            for w in conf.get("workers", []):
                try:
                    async with _session().get(w["url"].rstrip("/") + "/api/admin/sessions",
                                              timeout=aiohttp.ClientTimeout(total=3)) as r:
                        alive = {s.get("sessionid") for s in (await r.json(content_type=None)).get("data", {}).get("sessions", [])}
                except Exception:
                    continue
                for sid in [s for s, u in _where.items() if u == w["url"] and s not in alive]:
                    _where.pop(sid, None)
    app["gf_dispatch_task"] = asyncio.create_task(loop())


def setup(app):
    if os.getenv("GF_WORKER") == "1":                 # worker 自己不再分配（大家共用同一個 data/workers.json）
        return
    app.middlewares.append(dispatch_middleware)        # 在門禁（通關碼）之後才分配
    app.on_startup.append(_purge_loop)
