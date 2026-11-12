"""管理給外部網站用的 API 金鑰（data/api/keys.json，只存 sha256，原文只在建立時印一次）。

在 xiaoqing 容器裡跑（改完馬上生效，不用重開）：
  docker exec xiaoqing python /app/tools/api_keys.py create --name partner-a --origin https://a.example.com
  docker exec xiaoqing python /app/tools/api_keys.py list
  docker exec xiaoqing python /app/tools/api_keys.py disable --name partner-a
  docker exec xiaoqing python /app/tools/api_keys.py rotate  --name partner-a     # 換一把新的，舊的立刻失效

create 的選項：
  --origin        允許嵌入的網域，可以給好幾次（https://a.com）；沒給就不能放在任何網頁的 iframe 裡
  --max-sessions  同時幾條連線（預設 1；DGX 的 GPU 有限，別開太多）
  --rpm           每分鐘幾次 API（預設 60）
  --minutes       每條連線最長幾分鐘（預設 30）
  --no-chat       只能照念（echo），不能讓小晴自己回答（chat 會用到 LLM）
  --iai           允許用雲端 iAI 聲音（文字會送到學校 iAI 平台）
"""
import argparse
import hashlib
import json
import os
import re
import secrets
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_FILE = os.path.join(ROOT, "data", "api", "keys.json")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,31}$")
ORIGIN_RE = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d+)?$")


def load():
    try:
        with open(KEYS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"keys": {}}


def save(d):
    os.makedirs(os.path.dirname(KEYS_FILE), exist_ok=True)
    tmp = KEYS_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, KEYS_FILE)
    os.chmod(os.path.dirname(KEYS_FILE), 0o700)


def new_secret(name):
    raw = f"sk_{name}_{secrets.token_urlsafe(32)}"
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["create", "list", "disable", "enable", "rotate", "delete"])
    ap.add_argument("--name")
    ap.add_argument("--origin", action="append", default=[])
    ap.add_argument("--max-sessions", type=int, default=1)
    ap.add_argument("--rpm", type=int, default=60)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--iai", action="store_true")
    a = ap.parse_args()
    d = load()
    keys = d.setdefault("keys", {})

    if a.cmd == "list":
        if not keys:
            print("還沒有金鑰")
        for n, k in keys.items():
            print(f"{n:<20} {'啟用' if k.get('enabled', True) else '停用'}  網域 {', '.join(k.get('origins') or []) or '（無）'}"
                  f"  連線 {k.get('max_sessions', 1)}  每分鐘 {k.get('rpm', 60)}  每條 {k.get('minutes', 30)} 分"
                  f"  chat {'可' if k.get('chat', True) else '不可'}  iAI {'可' if k.get('iai') else '不可'}")
        return

    if not a.name or not NAME_RE.match(a.name):
        sys.exit("--name 要是小寫英文、數字、減號，2～32 個字（例如 partner-a）")
    if a.cmd == "create":
        if a.name in keys:
            sys.exit(f"{a.name} 已經有了；要換金鑰用 rotate")
        bad = [o for o in a.origin if not ORIGIN_RE.match(o.rstrip("/"))]
        if bad:
            sys.exit(f"網域格式不對：{bad}（要像 https://a.example.com，不要有路徑）")
        raw, h = new_secret(a.name)
        keys[a.name] = {"sha256": h, "enabled": True, "origins": [o.rstrip("/") for o in a.origin],
                        "max_sessions": max(1, a.max_sessions), "rpm": max(1, a.rpm), "minutes": max(1.0, a.minutes),
                        "chat": not a.no_chat, "iai": a.iai, "created": int(time.time())}
        save(d)
        print("建立好了。下面這把金鑰只會出現這一次，請用安全的方式交給對方（不要貼在聊天群組）：\n")
        print(raw)
        print("\n提醒對方：這把金鑰只能放在他的後端（伺服器），不能寫進網頁或 App。")
        return
    if a.name not in keys:
        sys.exit(f"沒有 {a.name}")
    if a.cmd == "rotate":
        raw, h = new_secret(a.name)
        keys[a.name]["sha256"] = h
        save(d)
        print("換好了，舊金鑰已失效。新的金鑰（只出現這一次）：\n")
        print(raw)
    elif a.cmd in ("disable", "enable"):
        keys[a.name]["enabled"] = a.cmd == "enable"
        save(d)
        print(f"{a.name} 已{'啟用' if a.cmd == 'enable' else '停用'}")
    elif a.cmd == "delete":
        keys.pop(a.name)
        save(d)
        print(f"{a.name} 已刪除")


if __name__ == "__main__":
    main()
