"""沒有人能錄音時：用 CosyVoice3 的「指令模式」從一段平常語氣的聲音，做出同一個音色的開心／溫柔／難過參考錄音，
給 BreezyVoice 當情緒參考。

在 xiaozhi-tts2 容器裡跑（那裡已經有 CosyVoice3 和模型）：
  docker exec xiaozhi-tts2 python /models/refs/make_emotion_refs.py \
      --src /models/refs/female_src.wav --out /models/refs/female --seconds 9.4

輸出：<out>/happy_1.wav、happy_1.txt …（每種情緒做 --takes 個版本，挑最好的）
"""
import argparse
import os
import sys
import types

import numpy as np
import soundfile as sf
import torch

LINES = {
    "happy": ("哇，你全部都寫完了耶！太厲害了，今天真的超有效率的！", "请非常开心地说一句话。"),
    "soft": ("沒關係，慢慢來就好。累了就先休息一下，我會在這裡陪你。", "Please say a sentence in a very soft voice."),
    "sad": ("嗯，這次考得不太理想，我知道你很失望。不過我們一起看看哪裡錯了。", "请非常伤心地说一句话。"),
}
SYS = "You are a helpful assistant."


def setup():
    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": k.get("weights_only", False)})
    import onnxruntime as ort

    class _S(ort.InferenceSession):
        def __init__(self, path, *a, providers=None, **k):
            avail = set(ort.get_available_providers())
            if providers:
                providers = [p for p in providers
                             if (p[0] if isinstance(p, (tuple, list)) else p) in avail] or ["CPUExecutionProvider"]
            super().__init__(path, *a, providers=providers, **k)
    ort.InferenceSession = _S
    for name in ("pyworld", "pyarrow"):
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)
            if name == "pyarrow":
                sys.modules["pyarrow.parquet"] = types.ModuleType("pyarrow.parquet")
    src = os.getenv("COSYVOICE_SRC", "/opt/CosyVoice")
    sys.path[:0] = [src, os.path.join(src, "third_party/Matcha-TTS")]
    from cosyvoice.cli.cosyvoice import AutoModel
    return AutoModel(model_dir=os.getenv("TTS2_MODEL_DIR", "/models/cosyvoice3"), fp16=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="平常語氣的聲音（wav）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0, help="只用前面幾秒（0 = 全部）")
    ap.add_argument("--takes", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    wav, sr = sf.read(a.src, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if a.seconds > 0:
        wav = wav[: int(a.seconds * sr)]
    src = os.path.join(a.out, "_src.wav")
    sf.write(src, wav, sr)

    cosy = setup()
    for emo, (text, ins) in LINES.items():
        for k in range(1, a.takes + 1):
            torch.manual_seed(1000 * k + len(emo))
            parts = [o["tts_speech"].squeeze(0).float().cpu().numpy()
                     for o in cosy.inference_instruct2(text, f"{SYS} {ins}<|endofprompt|>", src, stream=False)]
            audio = np.concatenate(parts)
            peak = float(np.abs(audio).max() or 1)
            audio = audio / peak * 0.9
            name = os.path.join(a.out, f"{emo}_{k}")
            sf.write(name + ".wav", audio, cosy.sample_rate)
            with open(name + ".txt", "w", encoding="utf-8") as f:
                f.write(text)
            print(f"做好 {name}.wav（{len(audio) / cosy.sample_rate:.1f} 秒）", flush=True)
    for p in os.listdir(a.out):
        os.chmod(os.path.join(a.out, p), 0o666)
    os.chmod(a.out, 0o777)


if __name__ == "__main__":
    main()
