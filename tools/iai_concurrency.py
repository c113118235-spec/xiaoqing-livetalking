"""量學校 iAI 語音「同時很多句」時會不會排隊（多人一起說話變慢，是不是卡在 iAI）。

在 xiaoqing 容器裡跑（那裡有 IAI_API_KEY；不會印出金鑰）：
  docker exec xiaoqing python -u /app/tools/iai_concurrency.py
  docker exec xiaoqing python -u /app/tools/iai_concurrency.py --levels 1,5,10,15 --voice 9D
看法：
  同時 1 句 約 X 秒；同時 15 句如果每句也約 X 秒 → iAI 能同時處理，慢在小晴這邊（對嘴／傳送）
  同時 15 句變成 X 的好幾倍、而且「最快～最慢」拉很開 → iAI 在排隊，要請 iAI 加資源或改用本機語音分擔
"""
import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                          # data/tts.json 用相對路徑
from tts.gftts import IAI_TTS_URL, iai_body  # noqa: E402

TEXT = "這題可以先把大問題拆成三個小步驟，我們一步一步來。"


def one(key, voice):
    t0 = time.time()
    try:
        r = requests.post(IAI_TTS_URL, timeout=60,
                          headers={"x-litellm-api-key": f"Bearer {key}", "accept": "application/json"},
                          json=iai_body(TEXT, voice=voice))
        ok = r.status_code == 200 and len(r.content) > 200
        return time.time() - t0, ok, "" if ok else f"{r.status_code} {r.text[:80]}"
    except Exception as e:
        return time.time() - t0, False, repr(e)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,5,10,15")
    ap.add_argument("--voice", default="9D")
    a = ap.parse_args()
    key = os.getenv("IAI_API_KEY", "")
    if not key:
        sys.exit("容器裡沒有 IAI_API_KEY")
    one(key, a.voice)                                   # 先暖一次，不算
    base = None
    for n in [int(x) for x in a.levels.split(",") if x.strip()]:
        t0 = time.time()
        with ThreadPoolExecutor(n) as ex:
            res = list(ex.map(lambda _: one(key, a.voice), range(n)))
        wall = time.time() - t0
        ts = [t for t, ok, _ in res if ok]
        errs = [e for _, ok, e in res if not ok]
        if not ts:
            print(f"同時 {n:2d} 句：全部失敗 {errs[:2]}")
            continue
        med = statistics.median(ts)
        base = base or med
        print(f"同時 {n:2d} 句：每句 中位數 {med:.2f}s、最快 {min(ts):.2f}s、最慢 {max(ts):.2f}s、全部做完 {wall:.2f}s"
              f"（是 1 句時的 {med / base:.1f} 倍）" + (f"｜失敗 {len(errs)}：{errs[0]}" if errs else ""), flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
