"""找出 iAI 語音（模型 Speaker）怎樣才會每次都是同一個人的聲音。

在 xiaoqing 容器裡跑（那裡有 IAI_API_KEY，也連得到學校）：
  docker exec xiaoqing python /app/tools/iai_probe.py
它會：
  1. 問 iAI 這個模型的資訊（不印金鑰）
  2. 用不同參數（voice、seed、speaker…）各生兩次同一句話，存到 data/iai_probe/
  3. 算每段的平均音高（Hz），同一組兩次差很多 = 聲音不固定；也印出 iAI 對不認得的參數回的錯誤訊息
"""
import json
import os
import sys
import time
from io import BytesIO

import numpy as np
import requests

BASE = os.getenv("IAI_BASE_URL", "https://www.iai.nkust.edu.tw/aihub/v1").rstrip("/")
MODEL = os.getenv("IAI_TTS_MODEL", "Speaker")
KEY = os.getenv("IAI_API_KEY", "")
H = {"x-litellm-api-key": f"Bearer {KEY}", "Authorization": f"Bearer {KEY}"}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "iai_probe")
TEXT = "嗨，我是小晴，今天也一起加油吧！"

VARIANTS = [
    ("default", {"voice": "default"}),
    ("default_x2", {"voice": "default"}),
    ("seed42", {"voice": "default", "seed": 42}),
    ("voice_42", {"voice": "42"}),
    ("voice_seed_42", {"voice": "seed_42"}),
    ("spk_female", {"voice": "female"}),
    ("spk_alloy", {"voice": "alloy"}),
    ("speaker_0", {"voice": "default", "speaker": 0}),
    ("lowtemp", {"voice": "default", "temperature": 0.1, "top_p": 0.5}),
    ("extra_body", {"voice": "default", "extra_body": {"seed": 42}}),
]


def info():
    for path in ("/model/info", "/models", "/audio/voices", "/audio/speech/voices"):
        url = BASE + path
        try:
            r = requests.get(url, headers=H, timeout=10)
        except Exception as e:
            print(f"GET {path} 失敗：{e}")
            continue
        print(f"GET {path} → {r.status_code}")
        if r.status_code != 200:
            print("   ", r.text[:300])
            continue
        try:
            d = r.json()
        except Exception:
            print("   ", r.text[:300])
            continue
        items = d.get("data", d) if isinstance(d, dict) else d
        if isinstance(items, list):
            for m in items:
                name = str(m.get("model_name") or m.get("id") or m)
                if path == "/models" and MODEL.lower() not in name.lower():
                    continue
                lp = dict(m.get("litellm_params") or {})
                for k in list(lp):                       # 不印任何金鑰／密碼
                    if any(s in k.lower() for s in ("key", "secret", "token", "password")):
                        lp[k] = "***"
                show = {"name": name, "litellm_params": lp, "model_info": m.get("model_info")}
                if MODEL.lower() in name.lower() or path != "/model/info":
                    print("   ", json.dumps(show, ensure_ascii=False, default=str)[:1500])
        else:
            print("   ", json.dumps(d, ensure_ascii=False)[:800])


def decode(data):
    try:
        import soundfile as sf
        w, sr = sf.read(BytesIO(data), dtype="float32", always_2d=True)
        return w.mean(axis=1), sr
    except Exception:
        import av
        out, sr = [], 24000
        with av.open(BytesIO(data)) as c:
            st = c.streams.audio[0]
            sr = st.rate or 24000
            rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
            for fr in c.decode(st):
                for f in rs.resample(fr):
                    out.append(f.to_ndarray().reshape(-1))
        return np.concatenate(out), sr


def pitch(w, sr):
    """很粗的平均音高：每 40ms 自相關找週期，只算有聲音的段落"""
    n = int(sr * 0.04)
    lo, hi = int(sr / 400), int(sr / 70)
    e_all = np.sqrt(np.mean(w ** 2)) + 1e-9
    f0 = []
    for i in range(0, len(w) - n, n // 2):
        x = w[i:i + n] - w[i:i + n].mean()
        if np.sqrt(np.mean(x ** 2)) < 0.5 * e_all:
            continue
        ac = np.correlate(x, x, "full")[n - 1:]
        if ac[0] <= 0:
            continue
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] / ac[0] > 0.45:
            f0.append(sr / k)
    return float(np.median(f0)) if f0 else 0.0


def main():
    if not KEY:
        sys.exit("沒有 IAI_API_KEY（要在 xiaoqing 容器裡跑）")
    os.makedirs(OUT, exist_ok=True)
    print("== 1. 模型資訊 ==")
    info()
    print(f"\n== 2. 同一句話各生兩次（{TEXT}）==")
    print(f"{'參數':<14}{'第1次Hz':>8}{'第2次Hz':>8}  {'差':>5}  說明")
    for name, extra in VARIANTS:
        hz, note = [], ""
        for k in (1, 2):
            body = {"model": MODEL, "input": TEXT, "response_format": "wav", **extra}
            t = time.time()
            try:
                r = requests.post(BASE + "/audio/speech", headers=H, json=body, timeout=40)
            except Exception as e:
                note = f"連線失敗 {e}"
                break
            if r.status_code != 200 or len(r.content) < 200:
                note = f"{r.status_code} {r.text[:200]}"
                break
            path = os.path.join(OUT, f"{name}_{k}.wav")
            w, sr = decode(r.content)
            import soundfile as sf
            sf.write(path, w, sr)
            hz.append(pitch(w, sr))
            note = f"{time.time() - t:.1f}s，{len(w) / sr:.1f}秒音訊"
        if len(hz) == 2:
            d = abs(hz[0] - hz[1])
            print(f"{name:<14}{hz[0]:>8.0f}{hz[1]:>8.0f}  {d:>5.0f}  {'穩定' if d < 15 else '不固定'}；{note}")
        else:
            print(f"{name:<14}{'-':>8}{'-':>8}  {'-':>5}  {note}")
    for p in os.listdir(OUT):
        os.chmod(os.path.join(OUT, p), 0o666)
    print(f"\n錄音在 {OUT}（每組 _1 _2 可以自己聽聽是不是同一個人）")


if __name__ == "__main__":
    main()
