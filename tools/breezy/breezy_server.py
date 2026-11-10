"""BreezyVoice（聯發科，台灣華語）語音服務：一男一女、帶情緒，OpenAI 相容的 /v1/audio/speech。

為什麼自己寫：官方 api.py 只能一個聲音，而且官方 Docker 是 x86 + CUDA 11.8，DGX Spark（ARM + 新顯卡）裝不起來。
這裡借用 xiaozhi-tts2（CosyVoice3）已經能跑的環境：BreezyVoice 本身就是 CosyVoice 改的，需要的套件幾乎一樣。

聲音（本人同意的參考錄音）放在 /voices：
  /voices/female/neutral.wav + neutral.txt     ← 必要（txt 是錄音逐字稿，越準越好）
  /voices/female/happy.wav   + happy.txt       ← 選用，沒有就用 neutral
  /voices/female/soft.wav    + soft.txt        ← 選用（溫柔）
  /voices/female/sad.wav     + sad.txt         ← 選用
  /voices/male/...                             ← 同上
  每段 5～15 秒、單人、安靜、不要背景音樂。

API：
  POST /v1/audio/speech  {"input": "...", "voice": "female|male", "emotion": "neutral|happy|soft|sad|angry", "speed": 1.0}
                         → audio/wav（22050 Hz）
  GET  /v1/voices        → 有哪些聲音、各有哪些情緒
  GET  /health

環境變數：
  BV_DIR       BreezyVoice 程式（預設 /bv）
  BV_MODEL     模型（預設 MediaTek-Research/BreezyVoice，第一次會下載到 HF 快取）
  VOICES_DIR   預設 /voices
  PORT         預設 8006
"""
from __future__ import annotations

import io
import os
import sys
import threading
import time
import types

BV_DIR = os.getenv("BV_DIR", "/bv")
BV_MODEL = os.getenv("BV_MODEL", "MediaTek-Research/BreezyVoice")
VOICES_DIR = os.getenv("VOICES_DIR", "/voices")
PORT = int(os.getenv("PORT", "8006"))
EMOTIONS = ("neutral", "happy", "soft", "sad")
PROMPT_MAX = float(os.getenv("BV_PROMPT_MAX", "9"))   # 參考錄音太長會拖慢每一句，超過就切在句子之間的停頓
AMP = os.getenv("BV_AMP", "bf16").lower()               # bf16（預設，較快）或 off
EMO_ALIAS = {"angry": "neutral", "gentle": "soft", "calm": "neutral", "": "neutral"}

sys.path.insert(0, BV_DIR)
sys.path.insert(0, os.path.join(BV_DIR, "third_party", "Matcha-TTS"))
os.environ.setdefault("PYTHONUTF8", "1")


_DIG = "零一二三四五六七八九"


def _int_zh(n: int) -> str:
    if n == 0:
        return "零"
    units = ["", "十", "百", "千"]
    big = ["", "萬", "億"]
    out, k = "", 0
    while n > 0 and k < len(big):
        part, n = n % 10000, n // 10000
        if part:
            s, z = "", False
            for i in range(3, -1, -1):
                d = part // (10 ** i) % 10
                if d:
                    if z:
                        s += "零"
                    s += _DIG[d] + units[i]
                    z = False
                elif s:
                    z = True
            out = s + big[k] + ("零" if out and not out.startswith("零") and part < 1000 else "") + out
        k += 1
    if out.startswith("一十"):
        out = out[1:]
    import re
    out = re.sub(r"(?<!十)二(?=[千百萬億])", "兩", out)     # 台灣念法：兩千、兩百、兩萬
    return out.rstrip("零")


def _simple_tn(text: str) -> str:
    """wetext 用不了時的簡易數字轉中文：12→十二、3.5→三點五、80%→百分之八十、2026年→二零二六年"""
    import re

    def year(m):
        return "".join(_DIG[int(c)] for c in m.group(1)) + "年"

    def pct(m):
        return "百分之" + num(m.group(1))

    def num(x):
        if "." in x:
            i, f = x.split(".", 1)
            return _int_zh(int(i or 0)) + "點" + "".join(_DIG[int(c)] for c in f)
        if len(x) > 1 and x.startswith("0"):
            return "".join(_DIG[int(c)] for c in x)
        return _int_zh(int(x))

    text = re.sub(r"(\d{4})年", year, text)
    text = re.sub(r"(\d+(?:\.\d+)?)%", pct, text)
    text = re.sub(r"\d+(?:\.\d+)?", lambda m: num(m.group(0)), text)
    return text


def _make_normalizers():
    """文字正規化（數字、單位念法）：
    1. /wetext 有 tts2 那邊複製來的 wetext 規則檔 → 直接用（不用連網）
    2. 否則試 wetext 預設下載（要連 modelscope.cn，實驗室網路可能連不到）
    3. 都不行 → 簡易數字轉中文"""
    zh = en = None
    wdir = os.getenv("WETEXT_DIR", "/wetext")
    try:
        from wetext import Normalizer as _WN
        if os.path.isfile(os.path.join(wdir, "zh", "tn", "tagger.fst")):
            zh = _WN(tagger_path=f"{wdir}/zh/tn/tagger.fst", verbalizer_path=f"{wdir}/zh/tn/verbalizer.fst", lang="zh")
            if os.path.isfile(os.path.join(wdir, "en", "tn", "tagger.fst")):
                en = _WN(tagger_path=f"{wdir}/en/tn/tagger.fst", verbalizer_path=f"{wdir}/en/tn/verbalizer.fst", lang="en")
            print(f"[breezy] 文字正規化：wetext（{wdir}）", flush=True)
        elif os.getenv("WETEXT_DOWNLOAD", "0") == "1":
            zh = _WN(lang="zh", operator="tn")
            en = _WN(lang="en", operator="tn")
            print("[breezy] 文字正規化：wetext（下載）", flush=True)
    except Exception as e:
        print(f"[breezy] wetext 用不了：{e}", flush=True)
    if zh is None:
        print("[breezy] 文字正規化：簡易數字轉中文", flush=True)

    class _Zh:
        def __init__(self, *a, **k):
            pass

        def normalize(self, text):
            if zh is not None:
                try:
                    return zh.normalize(text)
                except Exception:
                    pass
            return _simple_tn(text)

    class _En:
        def __init__(self, *a, **k):
            pass

        def normalize(self, text):
            if en is not None:
                try:
                    return en.normalize(text)
                except Exception:
                    pass
            return text

    return _Zh, _En


def _shims():
    """環境差異的補丁：
    - 沒有 WeTextProcessing（要 pynini，ARM 不好裝）→ 用 tts2 已經有的 wetext 頂替 tn.chinese / tn.english
    - onnxruntime 沒有 CUDA → 一律用 CPU（只用在參考錄音的前處理，快取後不影響速度）"""
    try:
        import tn.chinese.normalizer  # noqa: F401
    except Exception:
        _Zh, _En = _make_normalizers()
        tn = types.ModuleType("tn")
        zh = types.ModuleType("tn.chinese")
        zhn = types.ModuleType("tn.chinese.normalizer")
        en = types.ModuleType("tn.english")
        enn = types.ModuleType("tn.english.normalizer")
        zhn.Normalizer, enn.Normalizer = _Zh, _En
        sys.modules.update({"tn": tn, "tn.chinese": zh, "tn.chinese.normalizer": zhn,
                            "tn.english": en, "tn.english.normalizer": enn})

    # tts2 的 torchaudio 是精簡版（shim），補上 BreezyVoice 會用到但它沒有的東西
    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    try:
        import torchaudio.transforms as _tt
    except Exception:
        _tt = types.ModuleType("torchaudio.transforms")
        sys.modules["torchaudio.transforms"] = _tt
        torchaudio.transforms = _tt
    if not hasattr(_tt, "Resample"):
        import torch as _torch

        class _Resample(_torch.nn.Module):
            def __init__(self, orig_freq=16000, new_freq=16000, **k):
                super().__init__()
                self.o, self.n = int(orig_freq), int(new_freq)

            def forward(self, x):
                if self.o == self.n:
                    return x
                import librosa
                y = librosa.resample(x.detach().cpu().numpy(), orig_sr=self.o, target_sr=self.n)
                return _torch.from_numpy(y).to(x.dtype)

        _tt.Resample = _Resample
    print("[breezy] torchaudio：" + ", ".join(
        f"{n}={'有' if ok else '沒有'}" for n, ok in [
            ("load", hasattr(torchaudio, "load")),
            ("compliance.kaldi", __import__("importlib").util.find_spec("torchaudio.compliance.kaldi") is not None),
            ("transforms.Resample", hasattr(_tt, "Resample"))]), flush=True)

    import onnxruntime as ort
    _orig = ort.InferenceSession
    avail = set(ort.get_available_providers())

    def _session(*a, providers=None, **k):
        if providers:
            providers = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
        return _orig(*a, providers=providers, **k)

    ort.InferenceSession = _session


_shims()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from single_inference import CustomCosyVoice, get_bopomofo_rare  # noqa: E402

print(f"[breezy] 載入模型 {BV_MODEL} …", flush=True)
t0 = time.time()
cosy = CustomCosyVoice(BV_MODEL)
from g2pw import G2PWConverter  # noqa: E402

g2p = G2PWConverter()
print(f"[breezy] 模型就緒（{time.time() - t0:.0f} 秒）", flush=True)

_lock = threading.Lock()
_prompts: dict = {}          # (voice, emotion) → 參考錄音算好的特徵（快取）


def _load_16k(path: str) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    peak = float(np.abs(wav).max() or 1.0)
    if peak > 0.98:
        wav = wav / peak * 0.95
    return torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)


def _voices() -> dict:
    out = {}
    if not os.path.isdir(VOICES_DIR):
        return out
    for v in sorted(os.listdir(VOICES_DIR)):
        d = os.path.join(VOICES_DIR, v)
        if not os.path.isdir(d):
            continue
        emos = [e for e in EMOTIONS if os.path.isfile(os.path.join(d, e + ".wav"))]
        if "neutral" in emos:
            out[v] = emos
    return out


def _bopo(text: str) -> str:
    text = cosy.frontend.text_normalize_new(text, split=False)
    return get_bopomofo_rare(text, g2p)


def _trim_prompt(speech, text: str):
    """參考錄音超過 PROMPT_MAX 秒：在句號（。！？）對應的停頓處切開，只留前面幾句，逐字稿也跟著截。
    用字數比例估計每句結束的時間，再找附近最安靜的地方下刀。"""
    import re
    sr = 16000
    total = speech.shape[1] / sr
    if total <= PROMPT_MAX:
        return speech, text
    parts = [p for p in re.split(r"(?<=[。！？!?])", text) if p.strip()]
    if len(parts) < 2:
        return speech, text
    chars = [len(re.sub(r"\W", "", p)) or 1 for p in parts]
    allc = sum(chars)
    x = speech[0].numpy()
    hop = int(0.02 * sr)
    energy = np.array([float(np.sqrt(np.mean(x[i:i + hop] ** 2) + 1e-9)) for i in range(0, len(x) - hop, hop)])
    best = None
    acc = 0
    for i in range(len(parts) - 1):
        acc += chars[i]
        est = total * acc / allc
        if est > PROMPT_MAX or est < 3:
            continue
        lo, hi = int(max(0, est - 0.8) / 0.02), int(min(total, est + 0.8) / 0.02)
        if hi <= lo:
            continue
        j = lo + int(np.argmin(energy[lo:hi]))
        best = (j * 0.02 + 0.01, "".join(parts[:i + 1]))
    if not best:
        return speech, text
    cut, keep = best
    n = int(cut * sr)
    print(f"[breezy] 參考錄音 {total:.1f} 秒太長，切在 {cut:.1f} 秒：{keep}", flush=True)
    return speech[:, :n], keep


def _prompt(voice: str, emotion: str) -> dict:
    vs = _voices()
    if voice not in vs:
        voice = "female" if "female" in vs else (next(iter(vs)) if vs else None)
    if voice is None:
        raise HTTPException(503, f"{VOICES_DIR} 裡沒有任何聲音（至少要 female/neutral.wav + neutral.txt）")
    emotion = EMO_ALIAS.get(emotion, emotion)
    if emotion not in vs[voice]:
        emotion = "neutral"
    key = (voice, emotion)
    wav_path = os.path.join(VOICES_DIR, voice, emotion + ".wav")
    mtime = os.path.getmtime(wav_path)
    hit = _prompts.get(key)
    if hit and hit["mtime"] == mtime:
        return hit
    txt_path = os.path.join(VOICES_DIR, voice, emotion + ".txt")
    with open(txt_path, encoding="utf-8") as f:
        ptext = f.read().strip()
    speech = _load_16k(wav_path)
    speech, ptext = _trim_prompt(speech, ptext)
    mi = cosy.frontend.frontend_zero_shot("你好。", _bopo(ptext), speech)
    cached = {k: v for k, v in mi.items() if not k.startswith("text")}
    cached["mtime"] = mtime
    cached["voice"], cached["emotion"] = voice, emotion
    _prompts[key] = cached
    print(f"[breezy] 參考錄音就緒：{voice}/{emotion}（{speech.shape[1] / 16000:.1f} 秒）", flush=True)
    return cached


def synth(text: str, voice: str, emotion: str, speed: float = 1.0) -> bytes:
    import re
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "input 是空的")
    with _lock:
        p = _prompt(voice, emotion)
        pieces = []
        for sent in re.split(r"(?<=[？！。?!；;])\s*", text):
            if not sent.strip():
                continue
            tok, tok_len = cosy.frontend._extract_text_token(_bopo(sent))
            mi = {k: v for k, v in p.items() if k not in ("mtime", "voice", "emotion")}
            mi["text"], mi["text_len"] = tok, tok_len
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(AMP == "bf16")):
                out = cosy.model.inference(**mi)["tts_speech"]
            pieces.append(out.float().squeeze(0).numpy())
    audio = np.concatenate(pieces) if pieces else np.zeros(2205, np.float32)
    sr = 22050
    if abs(speed - 1.0) > 0.03:
        import librosa
        audio = librosa.effects.time_stretch(audio, rate=float(speed))
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class SpeechRequest(BaseModel):
    input: str
    voice: str = "female"
    emotion: str = "neutral"
    speed: float = 1.0
    model: str = ""
    response_format: str = "wav"


app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "voices": _voices()}


@app.get("/v1/voices")
def voices():
    return {"voices": _voices()}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    t = time.time()
    wav = synth(req.input, req.voice, req.emotion, req.speed)
    print(f"[breezy] {req.voice}/{req.emotion} {time.time() - t:.2f}s：{req.input[:40]}", flush=True)
    return Response(wav, media_type="audio/wav")


@app.exception_handler(Exception)
async def _err(_, e):
    if isinstance(e, HTTPException):
        return JSONResponse({"error": e.detail}, status_code=e.status_code)
    return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    for v, emos in _voices().items():          # 先把參考錄音算好，第一句話才不會慢
        for e in emos:
            try:
                _prompt(v, e)
            except Exception as ex:
                print(f"[breezy] {v}/{e} 參考錄音有問題：{ex}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
