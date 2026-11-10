"""從 Mozilla Common Voice 台灣華語（zh-TW，CC0 授權，志願者捐的錄音）挑出幾位男生（或女生），
各接成一段 8～10 秒的參考錄音，讓人試聽後挑一個當學伴的聲音。

資料：Hugging Face 上的 JacobLinCool/common_voice_19_0_zh-TW（test 分割，約 145 MB）
在 xiaoqing-indextts 容器裡跑：
  docker exec xiaoqing-indextts sh -c "pip install -q pyarrow && python /srv/pick_cv_voice.py --gender male --out /models/cv_pick/male"
輸出：<out>/cand_1.wav + cand_1.txt（逐字稿，BreezyVoice 要用）…、<out>/candidates.txt（說明）
"""
import argparse
import io
import os
import re

import numpy as np
import soundfile as sf

REPO = "JacobLinCool/common_voice_19_0_zh-TW"


def load_rows():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("data/test") and f.endswith(".parquet")]
    rows = []
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        rows += pq.read_table(p).to_pylist()
    return rows


def decode(audio) -> tuple:
    b = audio.get("bytes") if isinstance(audio, dict) else None
    if b:
        try:
            wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
            return wav.mean(axis=1), sr
        except Exception:                      # 舊版 soundfile 不能讀 mp3 → 交給 librosa（ffmpeg）
            import tempfile
            import librosa
            with tempfile.NamedTemporaryFile(suffix=".mp3") as t:
                t.write(b)
                t.flush()
                wav, sr = librosa.load(t.name, sr=None, mono=True)
            return wav.astype(np.float32), sr
    raise ValueError("沒有音訊")


def trim(wav, sr, thr=0.02):
    """去掉頭尾安靜的部分"""
    hop = int(sr * 0.02)
    e = np.array([np.sqrt(np.mean(wav[i:i + hop] ** 2)) for i in range(0, max(1, len(wav) - hop), hop)])
    if not len(e) or e.max() <= 0:
        return wav
    on = np.where(e > max(thr * e.max(), 1e-4))[0]
    if not len(on):
        return wav
    a, b = max(0, (on[0] - 3) * hop), min(len(wav), (on[-1] + 4) * hop)
    return wav[a:b]


def quality(wav, sr):
    """粗略的錄音品質：安靜處越安靜、沒有爆音越好"""
    hop = int(sr * 0.02)
    e = np.array([np.sqrt(np.mean(wav[i:i + hop] ** 2)) for i in range(0, max(1, len(wav) - hop), hop)]) + 1e-6
    snr = 20 * np.log10(np.percentile(e, 90) / np.percentile(e, 10))
    clip = float(np.mean(np.abs(wav) > 0.98))
    return snr - clip * 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="male", choices=["male", "female"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6, help="挑幾位候選")
    ap.add_argument("--seconds", type=float, default=9.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = load_rows()
    print(f"共 {len(rows)} 段錄音", flush=True)

    def is_gender(g):
        g = (g or "").lower()
        return g.startswith("female") if a.gender == "female" else (g.startswith("male") or g == "m")

    by = {}
    for r in rows:
        if not is_gender(r.get("gender")) or (r.get("down_votes") or 0) > 0 or (r.get("up_votes") or 0) < 2:
            continue
        by.setdefault(r["client_id"], []).append(r)
    speakers = sorted(by.items(), key=lambda kv: -len(kv[1]))
    print(f"符合條件的{'男' if a.gender == 'male' else '女'}生：{len(speakers)} 位", flush=True)

    cands = []
    for cid, rs in speakers[:40]:
        parts, texts, total, sr0 = [], [], 0.0, None
        for r in rs:
            try:
                wav, sr = decode(r["audio"])
            except Exception:
                continue
            if sr0 and sr != sr0:
                continue
            sr0 = sr
            wav = trim(wav, sr)
            if len(wav) < sr * 1.5:
                continue
            parts += [wav, np.zeros(int(sr * 0.35), np.float32)]
            texts.append(re.sub(r"\s+", "", r["sentence"]).rstrip("。！？!?") + "。")
            total += len(wav) / sr + 0.35
            if total >= a.seconds:
                break
        if total < 6 or not parts:
            continue
        audio = np.concatenate(parts[:-1])
        cands.append((quality(audio, sr0), cid, audio, sr0, "".join(texts), (rs[0].get("accents") or ""), (rs[0].get("age") or "")))

    cands.sort(key=lambda c: -c[0])
    lines = []
    for i, (q, cid, audio, sr, text, accent, age) in enumerate(cands[: a.n], 1):
        audio = audio / (np.abs(audio).max() or 1) * 0.9
        name = os.path.join(a.out, f"cand_{i}")
        sf.write(name + ".wav", audio, sr)
        with open(name + ".txt", "w", encoding="utf-8") as f:
            f.write(text)
        lines.append(f"cand_{i}：{len(audio) / sr:.1f} 秒，品質分數 {q:.0f}，年齡 {age or '未填'}，口音 {accent or '未填'}\n  {text}")
        print(lines[-1], flush=True)
    with open(os.path.join(a.out, "candidates.txt"), "w", encoding="utf-8") as f:
        f.write("來源：Mozilla Common Voice zh-TW（CC0），" + REPO + "\n\n" + "\n".join(lines) + "\n")
    for p in os.listdir(a.out):
        os.chmod(os.path.join(a.out, p), 0o666)
    os.chmod(a.out, 0o777)


if __name__ == "__main__":
    main()
