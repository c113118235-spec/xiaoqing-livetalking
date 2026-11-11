"""讀書黑板：在小晴輸入學號、密碼，由 DGX 登入高科大教學平台，讀這學期的作業、考試（只讀名稱和截止日）。

教學平台的登入流程（照它網頁的程式做）：
  1. GET  /api/v1/school/constant          → data.spbk：RSA 公鑰（Base64）
  2. 自己產生 AES 金鑰 ak、iv ai（各 12 個隨機位元組的 Base64，16 字），{"ak","ai"} 用 RSA（PKCS#1 v1.5）加密 → 標頭 X-WM-Sword
     （網頁寫的是 arrayBufferToBase64(getRandomValues(new Uint32Array(12)))，裡面 new Uint8Array(…) 會把 12 個數字
       各截成 1 個位元組，所以是 12 位元組 → 16 字 → 標準 AES-128）
  3. POST /api/v1/auth/login  username、password、captcha_enable 各自 Base64 後用 AES-CBC（ZeroPadding）加密
     → data.token；之後的請求帶標頭 X-WM-Authorization
  4. 跟書籤一樣：課程清單 → POST /courses/<id>/log 進入課程 → GET /assignment、/exam
  5. POST /api/v1/auth/logout，丟掉登入狀態
  ※ AES 照 CryptoJS 的做法實作（任意金鑰長度都跟 CryptoJS 一樣；16 字金鑰就是標準 AES-128）

安全：
  - 密碼只在這一次請求的記憶體裡用，不寫檔、不寫 log、不保存登入狀態
  - 只接受 HTTPS（Cloudflare 網址）或本機；區網 http 不收（EL_ALLOW_HTTP=1 才收）
  - 同一個學號或來源 10 分鐘內失敗 5 次就暫停 10 分鐘；同一個學號同時只跑一次
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import secrets
import time
import uuid
from datetime import date

import aiohttp
from aiohttp import web

from utils.logger import logger

BASE = os.getenv("EL_BASE", "https://elearning.nkust.edu.tw")
ALLOW_HTTP = os.getenv("EL_ALLOW_HTTP", "0") == "1"
UA = "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"

# ───────────── CryptoJS 相容的 AES（任意金鑰長度） ─────────────
_SBOX = [0] * 256
_INV = [0] * 256
_MUL2 = [0] * 256


def _init_tables():
    p = q = 1
    sbox = [0] * 256
    while True:
        p = p ^ ((p << 1) & 0xFF) ^ (0x1B if p & 0x80 else 0)
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        x = q ^ ((q << 1) | (q >> 7)) ^ ((q << 2) | (q >> 6)) ^ ((q << 3) | (q >> 5)) ^ ((q << 4) | (q >> 4))
        sbox[p] = (x ^ 0x63) & 0xFF
        if p == 1:
            break
    sbox[0] = 0x63
    for i in range(256):
        _SBOX[i] = sbox[i]
        _MUL2[i] = ((i << 1) ^ (0x1B if i & 0x80 else 0)) & 0xFF


_init_tables()
_RCON = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def _sub_word(t):
    return (_SBOX[(t >> 24) & 0xFF] << 24) | (_SBOX[(t >> 16) & 0xFF] << 16) | (_SBOX[(t >> 8) & 0xFF] << 8) | _SBOX[t & 0xFF]


def _key_schedule(key: bytes):
    if len(key) % 4:
        raise ValueError("金鑰長度要是 4 的倍數")
    nk = len(key) // 4
    words = [int.from_bytes(key[i:i + 4], "big") for i in range(0, len(key), 4)]
    rounds = nk + 6
    ks = []
    for r in range((rounds + 1) * 4):
        if r < nk:
            ks.append(words[r])
            continue
        t = ks[r - 1]
        if r % nk == 0:
            t = ((t << 8) | (t >> 24)) & 0xFFFFFFFF
            t = _sub_word(t) ^ (_RCON[r // nk] << 24)
        elif nk > 6 and r % nk == 4:
            t = _sub_word(t)
        ks.append(ks[r - nk] ^ t)
    return ks, rounds


def _encrypt_block(block: bytes, ks, rounds) -> bytes:
    s = [list(block[c * 4:c * 4 + 4]) for c in range(4)]       # s[col][row]

    def add(rk):
        for c in range(4):
            w = ks[rk * 4 + c]
            for r in range(4):
                s[c][r] ^= (w >> (24 - 8 * r)) & 0xFF

    add(0)
    for rnd in range(1, rounds + 1):
        for c in range(4):
            for r in range(4):
                s[c][r] = _SBOX[s[c][r]]
        t = [[s[(c + r) % 4][r] for r in range(4)] for c in range(4)]   # ShiftRows
        s[:] = t
        if rnd != rounds:
            for c in range(4):
                a = s[c]
                x = a[0] ^ a[1] ^ a[2] ^ a[3]
                b0 = a[0]
                a[0] ^= x ^ _MUL2[a[0] ^ a[1]]
                a[1] ^= x ^ _MUL2[a[1] ^ a[2]]
                a[2] ^= x ^ _MUL2[a[2] ^ a[3]]
                a[3] ^= x ^ _MUL2[a[3] ^ b0]
        add(rnd)
    return bytes(s[c][r] for c in range(4) for r in range(4))


def cryptojs_aes_cbc_zero(plain: bytes, key: bytes, iv: bytes) -> str:
    """等同 CryptoJS.AES.encrypt(Utf8, key, {iv, mode: CBC, padding: ZeroPadding}).toString()"""
    ks, rounds = _key_schedule(key)
    if len(plain) % 16:
        plain += b"\0" * (16 - len(plain) % 16)
    prev = iv[:16].ljust(16, b"\0")
    out = b""
    for i in range(0, len(plain), 16):
        blk = bytes(a ^ b for a, b in zip(plain[i:i + 16], prev))
        prev = _encrypt_block(blk, ks, rounds)
        out += prev
    return base64.b64encode(out).decode()


def wm_field(value: str, ak: str, ai: str) -> str:
    """教學平台的 secureAesEncrypt：先 Base64（UTF-8），再 AES-CBC ZeroPadding"""
    if not value or not value.strip():
        return value
    b64 = base64.b64encode(value.encode("utf-8"))
    return cryptojs_aes_cbc_zero(b64, ak.encode(), ai.encode())


def rsa_encrypt(public_key_text: str, msg: str) -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    pem = public_key_text.strip()
    if "BEGIN" not in pem:
        pem = "-----BEGIN PUBLIC KEY-----\n" + "\n".join(pem[i:i + 64] for i in range(0, len(pem), 64)) + "\n-----END PUBLIC KEY-----"
    key = serialization.load_pem_public_key(pem.encode())
    return base64.b64encode(key.encrypt(msg.encode(), padding.PKCS1v15())).decode()


# ───────────── 登入＋讀作業 ─────────────
class ElError(Exception):
    def __init__(self, msg, status=400, login=False):
        super().__init__(msg)
        self.status, self.login = status, login


async def _json(resp):
    try:
        return await resp.json(content_type=None)
    except Exception:
        return {}


async def collect(username: str, password: str) -> dict:
    ak = base64.b64encode(secrets.token_bytes(12)).decode()      # 16 字，跟網頁一樣
    ai = base64.b64encode(secrets.token_bytes(12)).decode()
    hdr = {"User-Agent": UA, "Accept": "application/json, text/plain, */*", "Referer": BASE + "/moocs/",
           "Origin": BASE, "Accept-Language": "zh-TW,zh;q=0.9"}
    timeout = aiohttp.ClientTimeout(total=90, sock_connect=10, sock_read=30)
    async with aiohttp.ClientSession(headers=hdr, timeout=timeout, cookie_jar=aiohttp.CookieJar()) as s:
        async with s.get(BASE + "/api/v1/school/constant") as r:
            d = await _json(r)
        spbk = (d.get("data") or {}).get("spbk")
        if not spbk:
            raise ElError("教學平台沒有回傳加密公鑰（可能在維護）", 502)
        pub = base64.b64decode(spbk).decode("utf-8", "ignore")
        sword = rsa_encrypt(pub, json.dumps({"ak": ak, "ai": ai}, separators=(",", ":")))
        body = {"username": wm_field(username, ak, ai), "password": wm_field(password, ak, ai),
                "captcha_uuid": str(uuid.uuid4()), "captcha_code": "", "captcha_enable": wm_field("false", ak, ai)}
        async with s.post(BASE + "/api/v1/auth/login", json=body, headers={"X-WM-Sword": sword}) as r:
            d = await _json(r)
            status = r.status
        token = (d.get("data") or {}).get("token") if isinstance(d.get("data"), dict) else None
        if status != 200 or not token:
            msg = str(d.get("message") or d.get("error") or d.get("msg") or "")[:80]
            raise ElError("學號或密碼不對" + (f"（{msg}）" if msg else ""), 401, login=True)
        auth = {"X-WM-Authorization": token}
        try:
            return await _read(s, auth)
        finally:
            try:
                async with s.post(BASE + "/api/v1/auth/logout", headers=auth) as r:
                    await r.read()
            except Exception:
                pass


async def _read(s, auth) -> dict:
    async def get(path):
        async with s.get(BASE + path, headers=auth) as r:
            if r.status in (401, 403):
                raise ElError("教學平台登入失效", 401, login=True)
            if r.status != 200:
                raise ElError(f"{path.split('?')[0]} {r.status}", 502)
            return await _json(r)

    def list_of(d):
        x = d.get("data")
        if isinstance(x, dict) and isinstance(x.get("list"), list):
            return x["list"]
        return x if isinstance(x, list) else []

    cs = []
    for off in range(0, 300, 15):
        d = await get(f"/api/v1/courses?offset={off}&size=15&purpose=learn&sort=new&apply_status=registered&direction=desc&keyword=")
        page = d.get("data") if isinstance(d.get("data"), list) else []
        cs += page
        if len(page) < 15:
            break

    def sem(c):
        m = re.match(r"^(\d{4})_", str(c.get("name", "")))
        return m.group(1) if m else ""

    cur = max([sem(c) for c in cs if sem(c)] or [""])
    target = [c for c in cs if sem(c) == cur and c.get("my_role") != "teacher"]

    def short(n):
        n = re.sub(r"^\d+_\d+_", "", str(n or ""))
        return re.sub(r"_.*$", "", re.sub(r"[(（].*$", "", n)).strip()

    def clean(t):
        return re.sub(r"\s+", " ", re.sub(r"【([^】]*)】", r"\1 ", str(t or ""))).strip()

    def done(x):
        return bool((x.get("answer_process") and x.get("answer_process") != "none")
                    or x.get("status") in ("done", "finished", "submitted", "complete") or x.get("score") is not None)

    items = []
    for c in target:
        async with s.post(BASE + f"/api/v1/courses/{c['course_id']}/log", headers=auth, json={}) as r:
            await r.read()
        for kind in ("assignment", "exam"):
            try:
                lst = list_of(await get("/api/v1/" + kind))
            except ElError as e:
                if e.login:
                    raise
                lst = []
            for x in lst:
                if str(x.get("course_id")) != str(c["course_id"]):
                    continue
                close = str(x.get("answer_close_time") or x.get("close_time") or x.get("end_time") or "")
                due = close[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", close) and not close.startswith("9999") else ""
                exam = kind == "exam" or bool(re.search(r"exam|test|quiz", str(x.get("type", "")), re.I))
                items.append({"k": "el:" + str(x.get("exam_id") or x.get("id") or f"{c['course_id']}:{x.get('title')}"),
                              "text": (short(c.get("name")) + "｜" + ("考試 " if exam else "") + clean(x.get("title")))[:60],
                              "due": due, "done": done(x)})
    today = date.today()

    def keep(t):
        if not t["due"]:
            return True
        n = (date.fromisoformat(t["due"]) - today).days
        return n >= 0 if t["done"] else n >= -14

    return {"v": 1, "at": int(time.time() * 1000), "sem": cur, "items": [t for t in items if keep(t)]}


# ───────────── 小晴的 API ─────────────
_fails: dict = {}          # key → [時間, …]
_running: set = set()
FAIL_MAX, FAIL_WIN, LOCK = 5, 600, 600


def _locked(*keys) -> int:
    now = time.time()
    worst = 0
    for k in keys:
        lst = [t for t in _fails.get(k, []) if now - t < FAIL_WIN + LOCK]
        _fails[k] = lst
        recent = [t for t in lst if now - t < FAIL_WIN]
        if len(recent) >= FAIL_MAX:
            worst = max(worst, int(LOCK - (now - recent[-1])))
    return worst


def _fail(*keys):
    for k in keys:
        _fails.setdefault(k, []).append(time.time())


def _secure(request) -> bool:
    if ALLOW_HTTP:
        return True
    peer = (request.remote or "")
    proto = request.headers.get("X-Forwarded-Proto", "") or request.headers.get("CF-Visitor", "")
    return "https" in proto or peer in ("127.0.0.1", "::1")


def _mask(u: str) -> str:
    return u[:3] + "***" + u[-2:] if len(u) > 5 else "***"


async def sync(request):
    if not _secure(request):
        return web.json_response({"ok": False, "error": "為了保護密碼，只能從 https 的小晴網址使用這個功能"}, status=403)
    try:
        body = await request.json()
    except Exception:
        body = {}
    username = str(body.get("username", "")).strip().upper()
    password = str(body.get("password", ""))
    body = None
    if not re.match(r"^[A-Z0-9]{4,20}$", username) or not (0 < len(password) <= 64):
        return web.json_response({"ok": False, "error": "請輸入學號和密碼"}, status=400)
    ip = request.headers.get("CF-Connecting-IP") or request.remote or "?"
    wait = _locked("u:" + username, "ip:" + ip)
    if wait > 0:
        return web.json_response({"ok": False, "error": f"失敗太多次，請 {max(1, wait // 60)} 分鐘後再試"}, status=429)
    if username in _running:
        return web.json_response({"ok": False, "error": "這個學號正在同步中，請稍等"}, status=409)
    _running.add(username)
    t0 = time.time()
    try:
        pack = await asyncio.wait_for(collect(username, password), timeout=120)
        logger.info(f"[gf-elearn] {_mask(username)} 同步完成：{len(pack['items'])} 項（{time.time() - t0:.1f} 秒）")
        return web.json_response({"ok": True, "pack": pack})
    except ElError as e:
        if e.login:
            _fail("u:" + username, "ip:" + ip)
        logger.warning(f"[gf-elearn] {_mask(username)} 同步失敗：{e}")
        return web.json_response({"ok": False, "error": str(e)}, status=e.status)
    except asyncio.TimeoutError:
        return web.json_response({"ok": False, "error": "教學平台太慢了，請稍後再試"}, status=504)
    except Exception as e:
        logger.warning(f"[gf-elearn] {_mask(username)} 同步出錯：{type(e).__name__}")
        return web.json_response({"ok": False, "error": "同步出錯了，請稍後再試"}, status=500)
    finally:
        password = None
        _running.discard(username)


def setup(app):
    app.router.add_post("/el/sync", sync)
    logger.info("[gf-elearn] 教學平台同步（帳密不保存）")
