"""比較幾個 LLM 當小晴的大腦：開口快不快（第一個字幾秒）、會不會冒簡體字、有沒有照格式加情緒標籤。

在 xiaoqing 容器裡跑（用容器裡的 IAI_API_KEY，不會印出金鑰）：
  docker exec xiaoqing python -u /app/tools/llm_compare.py
  docker exec xiaoqing python -u /app/tools/llm_compare.py --models qwen-3.8-27b,Qwen3.8-Flash-Next
實驗室專用平台（網址和金鑰從 data/llm.json 讀，先照 README 建好這個檔）：
  docker exec xiaoqing python -u /app/tools/llm_compare.py --lab
  docker exec xiaoqing python -u /app/tools/llm_compare.py --lab --models qwen-3.8-27b,Qwen3.8-Flash-Next
"""
import argparse
import os
import statistics
import sys
import time

from openai import OpenAI

LAB_DEFAULT = "qwen-3.8-27b,Qwen3.8-Flash-Next,nemotron-omni-30b,nemotron-3-super-120b,nvidia-nemotron-3-ultra-550b-a55b"
DEFAULT = "Nkust,qwen-3.8-27b,Qwen3.8-Flash-Next,nemotron-omni-30b,nemotron-3-super-120b,nvidia-nemotron-3-ultra-550b-a55b"
SYSTEM = ("你是「小晴」，是使用者的 AI 學伴，一律用台灣繁體中文回答，絕對不要用簡體字。"
          "回答要像面對面說話一樣口語、自然、簡短，一次最多三句話。不要使用表情符號、條列或 Markdown。"
          "每一句話最前面都要加一個情緒標籤，只能從【開心】【難過】【生氣】【溫柔】【平常】裡選一個。")
QUESTIONS = ["嗨小晴，今天好累喔，作業好多寫不完。",
             "你可以用簡單的話跟我解釋什麼是機器學習嗎？",
             "我明天要考微積分，好緊張，怎麼辦？"]
# 常見的簡體字（出現就代表模型沒乖乖用繁體）
SIMPLIFIED = set("这个们说时会对还没来过为么样实学习题后发现进开关问间应该让给经长东车门见听写书爱乐专业务师网钱")


def run(client, model, q, no_think):
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if no_think else {}
    t0, first, out = time.time(), None, []
    stream = client.chat.completions.create(model=model, stream=True, temperature=0.7, max_tokens=200,
                                            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                            extra_body=extra or None)
    for ch in stream:
        d = ch.choices[0].delta.content if ch.choices else None
        if d:
            if first is None:
                first = time.time() - t0
            out.append(d)
    return first, time.time() - t0, "".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--lab", action="store_true", help="用 data/llm.json 的網址和金鑰（實驗室專用平台）")
    ap.add_argument("--base-url", default=os.getenv("IAI_BASE_URL", "https://www.iai.nkust.edu.tw/aihub/v1"))
    ap.add_argument("--key-env", default="IAI_API_KEY")
    ap.add_argument("--think", action="store_true", help="不關閉思考模式（預設關掉，開口才快）")
    a = ap.parse_args()
    if a.lab:
        import json
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            with open(os.path.join(root, "data", "llm.json"), encoding="utf-8") as f:
                c = json.load(f)
        except OSError:
            sys.exit("找不到 data/llm.json")
        a.base_url = c.get("base_url", "").rstrip("/")
        key = c.get("api_key") or os.getenv(str(c.get("key_env") or ""), "")
        a.models = a.models or LAB_DEFAULT
    else:
        key = os.getenv(a.key_env, "")
        a.models = a.models or DEFAULT
    if not key:
        sys.exit("沒有金鑰（data/llm.json 的 api_key，或容器裡的環境變數）")
    client = OpenAI(api_key=key, base_url=a.base_url, timeout=60,
                    default_headers={"x-litellm-api-key": f"Bearer {key}"})
    summary = []
    for m in [x.strip() for x in a.models.split(",") if x.strip()]:
        print(f"\n===== {m} =====", flush=True)
        firsts, totals, simp, tagged, ok = [], [], 0, 0, 0
        for q in QUESTIONS:
            try:
                f, t, text = run(client, m, q, not a.think)
            except Exception as e:
                print(f"  失敗：{str(e)[:150]}")
                continue
            ok += 1
            firsts.append(f or t)
            totals.append(t)
            s = sum(1 for c in text if c in SIMPLIFIED)
            simp += s
            tagged += text.startswith("【")
            print(f"  第一個字 {f or t:.2f}s、全部 {t:.2f}s、{len(text)} 字" + (f"、簡體字 {s} 個" if s else ""))
            print(f"    {text[:120]}")
        if ok:
            summary.append((m, statistics.median(firsts), statistics.median(totals), simp, f"{tagged}/{ok}"))
    print("\n總結（第一個字越快 = 小晴越快開口）")
    print(f"{'模型':40s} 第一個字  全部    簡體字  有情緒標籤")
    for m, f, t, s, tg in sorted(summary, key=lambda x: x[1]):
        print(f"{m:40s} {f:6.2f}s {t:6.2f}s {s:5d}   {tg}")


if __name__ == "__main__":
    main()
