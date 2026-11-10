"""IndexTTS-2.5（嗶哩嗶哩）語音服務：音色和口音照台灣的參考錄音，情緒用 8 個數字另外控制，不需要情緒錄音。
介面跟 tools/breezy 一樣（OpenAI 相容 /v1/audio/speech，多 voice、emotion），可以直接互換。

授權：bilibili Model Use License Agreement（https://github.com/index-tts/index-tts/blob/main/LICENSE），實驗室研究使用。

聲音：跟 BreezyVoice 共用 /voices/<voice>/neutral.wav（只要平常語氣那段）。
情緒：EMO_VEC（順序：開心、生氣、難過、害怕、厭惡、憂鬱、驚訝、平靜），IDX_EMO_ALPHA 調整整體強度。

台灣念法：
  - 繁體先轉簡體（模型主要用簡體訓練，繁體字容易念錯）
  - 數字照台灣念法轉中文（兩千、兩百）
  - TW_READINGS 裡的詞用 IndexTTS 的讀音標註 <字|拼音> 強制台灣讀音（例如 垃圾 ㄌㄜˋ ㄙㄜˋ）

API：
  POST /v1/audio/speech {"input": "...", "voice": "female|male", "emotion": "neutral|happy|soft|sad|angry"}
  GET  /v1/voices、/health

環境變數：IDX_DIR（程式，預設 /idx）、IDX_MODEL_DIR（模型，預設 /models/indextts25）、VOICES_DIR、PORT（預設 8007）、
         IDX_EMO_ALPHA（預設 0.8）、IDX_BF16（預設 1）
"""
from __future__ import annotations

import io
import os
import re
import sys
import threading
import time
import types

IDX_DIR = os.getenv("IDX_DIR", "/idx")
MODEL_DIR = os.getenv("IDX_MODEL_DIR", "/models/indextts25")
VOICES_DIR = os.getenv("VOICES_DIR", "/voices")
PORT = int(os.getenv("PORT", "8007"))
EMO_ALPHA = float(os.getenv("IDX_EMO_ALPHA", "0.8"))
BF16 = os.getenv("IDX_BF16", "1") == "1"
sys.path.insert(0, IDX_DIR)

#            開心  生氣  難過  害怕  厭惡  憂鬱  驚訝  平靜
EMO_VEC = {
    "happy": [0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0],
    "soft": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
    "sad": [0.0, 0.0, 0.55, 0.0, 0.0, 0.25, 0.0, 0.0],
    "angry": [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
EMO_ALIAS = {"gentle": "soft", "calm": "neutral", "": "neutral"}

# 台灣常見讀音（模型預設是大陸念法）；格式：詞 → 每個字的拼音（數字是聲調）
TW_READINGS = {          # key 用簡體（文字已先轉成簡體）
    "垃圾": ["le4", "se4"],
    "星期": ["xing1", "qi2"],
    "期待": ["qi2", "dai4"],
    "危险": ["wei2", "xian3"],
    "危机": ["wei2", "ji1"],
    "头发": ["tou2", "fa3"],
    "认识": ["ren4", "shi4"],
    "知识": ["zhi1", "shi4"],
    "企业": ["qi4", "ye4"],
    "质量": ["zhi2", "liang4"],
}


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




def _shims():
    import torch
    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": k.get("weights_only", False)})
    # 文字正規化（WeTextProcessing 要 pynini，ARM 不好裝）→ 替身；我們自己先做數字轉換，再關掉模型的正規化
    class _N:
        def __init__(self, *a, **k):
            pass

        def normalize(self, text):
            return _simple_tn(text)
    for name in ("tn", "tn.chinese", "tn.english"):
        sys.modules.setdefault(name, types.ModuleType(name))
    zh = types.ModuleType("tn.chinese.normalizer")
    en = types.ModuleType("tn.english.normalizer")
    zh.Normalizer = en.Normalizer = _N
    sys.modules["tn.chinese.normalizer"], sys.modules["tn.english.normalizer"] = zh, en
    _patch_torchaudio()


def _patch_torchaudio():
    """tts2 的 torchaudio 是精簡版（shim），補上 IndexTTS 會用到但它沒有的東西"""
    import math

    import torch
    import torchaudio
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    try:
        import torchaudio.functional as F
    except Exception:
        F = types.ModuleType("torchaudio.functional")
        sys.modules["torchaudio.functional"] = F
        torchaudio.functional = F

    # torchaudio.functional.functional._hz_to_mel / _mel_to_hz（照 torchaudio 原始碼）
    def _hz_to_mel(freq, mel_scale="htk"):
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + (freq / 700.0))
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz, logstep = 1000.0, math.log(6.4) / 27.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def _mel_to_hz(mels, mel_scale="htk"):
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz, logstep = 1000.0, math.log(6.4) / 27.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
        return freqs

    if "torchaudio.functional.functional" not in sys.modules:
        try:
            import torchaudio.functional.functional  # noqa: F401
        except Exception:
            ff = types.ModuleType("torchaudio.functional.functional")
            ff._hz_to_mel, ff._mel_to_hz = _hz_to_mel, _mel_to_hz
            sys.modules["torchaudio.functional.functional"] = ff
            F.functional = ff

    if not hasattr(F, "resample"):
        def resample(x, orig_freq, new_freq, **k):
            if int(orig_freq) == int(new_freq):
                return x
            import librosa
            y = librosa.resample(x.detach().cpu().numpy(), orig_sr=int(orig_freq), target_sr=int(new_freq))
            return torch.from_numpy(y).to(x.dtype)
        F.resample = resample

    try:
        import torchaudio.transforms as T
    except Exception:
        T = types.ModuleType("torchaudio.transforms")
        sys.modules["torchaudio.transforms"] = T
        torchaudio.transforms = T
    if not hasattr(T, "Resample"):
        class Resample(torch.nn.Module):
            def __init__(self, orig_freq=16000, new_freq=16000, **k):
                super().__init__()
                self.o, self.n = orig_freq, new_freq

            def forward(self, x):
                return F.resample(x, self.o, self.n)
        T.Resample = Resample
    if not hasattr(T, "MelSpectrogram"):
        class MelSpectrogram(torch.nn.Module):
            def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None, f_min=0.0,
                         f_max=None, n_mels=128, power=2.0, center=True, norm=None, mel_scale="htk", **k):
                super().__init__()
                import librosa
                self.n_fft, self.win = n_fft, win_length or n_fft
                self.hop, self.power, self.center = hop_length or self.win // 2, power, center
                fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min,
                                         fmax=f_max, htk=(mel_scale == "htk"), norm=norm)
                self.register_buffer("fb", torch.from_numpy(fb).float(), persistent=False)
                self.register_buffer("window", torch.hann_window(self.win), persistent=False)

            def forward(self, x):
                spec = torch.stft(x, self.n_fft, self.hop, self.win, self.window.to(x.device), center=self.center,
                                  return_complex=True).abs() ** self.power
                return torch.matmul(self.fb.to(x.device), spec)
        T.MelSpectrogram = MelSpectrogram


_shims()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

try:
    import opencc
    _t2s = opencc.OpenCC("t2s").convert
except Exception:
    _t2s = lambda s: s  # noqa: E731

from indextts.infer_v2_5 import IndexTTS2  # noqa: E402

print(f"[idx] 載入 IndexTTS-2.5（{MODEL_DIR}）…", flush=True)
_t0 = time.time()
tts = IndexTTS2(cfg_path=os.path.join(MODEL_DIR, "config.yaml"), model_dir=MODEL_DIR,
                use_bf16=BF16, use_cuda_kernel=False, use_deepspeed=False)
print(f"[idx] 模型就緒（{time.time() - _t0:.0f} 秒）", flush=True)
_lock = threading.Lock()


def _voices() -> dict:
    out = {}
    if os.path.isdir(VOICES_DIR):
        for v in sorted(os.listdir(VOICES_DIR)):
            if os.path.isfile(os.path.join(VOICES_DIR, v, "neutral.wav")):
                out[v] = ["neutral", *EMO_VEC]
    return out


def _prompt_path(voice: str) -> str:
    """參考錄音超過 12 秒就只取前 12 秒（另存一份），太長只會變慢"""
    src = os.path.join(VOICES_DIR, voice, "neutral.wav")
    info = sf.info(src)
    if info.duration <= 12:
        return src
    dst = f"/tmp/idx_prompt_{voice}.wav"
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        wav, sr = sf.read(src, dtype="float32", always_2d=True)
        sf.write(dst, wav[: int(12 * sr)].mean(axis=1), sr)
    return dst


def _tw(text: str) -> str:
    text = _t2s(_simple_tn(text.strip()))
    for word, py in sorted(TW_READINGS.items(), key=lambda x: -len(x[0])):
        if word in text and len(word) == len(py):
            text = text.replace(word, "".join(f"<{c}|{p}>" for c, p in zip(word, py)))
    return text


def synth(text: str, voice: str, emotion: str) -> bytes:
    if not text.strip():
        raise HTTPException(400, "input 是空的")
    vs = _voices()
    if not vs:
        raise HTTPException(503, f"{VOICES_DIR} 裡沒有聲音（至少要 female/neutral.wav）")
    if voice not in vs:
        voice = "female" if "female" in vs else next(iter(vs))
    emotion = EMO_ALIAS.get(emotion, emotion)
    kw = {}
    if emotion in EMO_VEC:
        kw = {"emo_vector": tts.normalize_emo_vec(EMO_VEC[emotion]), "emo_alpha": EMO_ALPHA}
    with _lock:
        out = tts.infer(spk_audio_prompt=_prompt_path(voice), text=_tw(text), output_path=None, lang="ZH",
                        text_normalization=False, verbose=False, **kw)
    if not out:
        raise HTTPException(500, "沒有產生聲音")
    sr, wav = out
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.reshape(-1) if wav.shape[-1] == 1 else wav.mean(axis=-1)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
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
    return {"ok": True, "engine": "indextts-2.5", "voices": _voices()}


@app.get("/v1/voices")
def voices():
    return {"voices": _voices()}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    t = time.time()
    wav = synth(req.input, req.voice, req.emotion)
    print(f"[idx] {req.voice}/{req.emotion} {time.time() - t:.2f}s：{req.input[:40]}", flush=True)
    return Response(wav, media_type="audio/wav")


@app.exception_handler(Exception)
async def _err(_, e):
    if isinstance(e, HTTPException):
        return JSONResponse({"error": e.detail}, status_code=e.status_code)
    return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    try:                                   # 暖機：第一句才不會特別慢
        for v in _voices():
            synth("你好。", v, "neutral")
            print(f"[idx] 參考錄音就緒：{v}", flush=True)
    except Exception as e:
        print(f"[idx] 暖機失敗：{e}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
