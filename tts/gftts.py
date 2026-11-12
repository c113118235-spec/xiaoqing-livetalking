"""小晴的情緒語音。

- 依序試 TW_TTS_URLS：IndexTTS-2.5（8007，tools/indextts）→ BreezyVoice（8006，tools/breezy）
  兩個都用開始畫面選的聲音（female / male）＋情緒
- 都失敗 → tts2（CosyVoice3，8005）→ 再失敗 → tts（Breeze2-VITS，8002）
- 學校 iAI 平台的語音（/aihub/v1/audio/speech，模型 Speaker）可以排在最前面試用：
  開始畫面選「iAI 9D」「iAI 36D」就用那個固定的說話者（失敗回到本機語音）；
  或 data/tts.json 寫 {"order": ["iai", "local"]}，不管選哪個聲音都先用 iAI，失敗才回到上面的本機語音；
  不用重開，每句都會讀這個檔（改了馬上生效）。iAI 目前只有一個聲音、不吃情緒參數。
  金鑰用跟 LLM 一樣的 IAI_API_KEY；IAI_TTS_MODEL（預設 Speaker）、IAI_TTS_VOICE（預設 default）
- 情緒：llm.py 把【開心】等標籤轉成 datainfo['tts']['emotion']
- TW_TTS_URLS 設成空字串就跳過台灣華語語音；想先用 BreezyVoice 就把順序對調
- 網址可用 --TTS_SERVER 指定 tts2，預設 http://127.0.0.1:8005（LiveTalking 用 --network host 跑，所以連得到）

用法：python app.py ... --tts gftts
"""
import os
import time
from io import BytesIO

import numpy as np
import requests
import resampy
import soundfile as sf

from registry import register
from utils.logger import logger

from .base_tts import BaseTTS, State

TW_TTS_URLS = [u.strip() for u in os.getenv(
    "TW_TTS_URLS", "http://127.0.0.1:8007/v1/audio/speech,http://127.0.0.1:8006/v1/audio/speech").split(",") if u.strip()]
TTS2_URL = os.getenv("TTS2_URL", "http://127.0.0.1:8005/v1/audio/speech")
TTS_URL = os.getenv("TTS_URL", "http://127.0.0.1:8002/v1/audio/speech")     # 備援：Breeze2-VITS
TIMEOUT = float(os.getenv("TTS2_TIMEOUT", "15"))
IAI_TTS_URL = os.getenv("IAI_TTS_URL", os.getenv("IAI_BASE_URL", "https://www.iai.nkust.edu.tw/aihub/v1").rstrip("/") + "/audio/speech")
IAI_TTS_MODEL = os.getenv("IAI_TTS_MODEL", "Speaker")
IAI_TTS_VOICE = os.getenv("IAI_TTS_VOICE", "default")
IAI_TTS_VOICES = os.getenv("IAI_TTS_VOICES", "9D,36D")          # 開始畫面列出的 iAI 聲音（固定的說話者）
IAI_NAME_RE = __import__("re").compile(r"^[A-Za-z0-9-]{1,24}$")
IAI_TTS_TIMEOUT = float(os.getenv("IAI_TTS_TIMEOUT", "20"))
TTS_CONF = os.getenv("TTS_CONF", os.path.join("data", "tts.json"))
_conf = {"mtime": None, "order": ["local"], "iai": {}}


def _load_conf():
    """data/tts.json（改了馬上生效）：
      "order": ["iai", "local"]   不管選哪個聲音都先用 iAI（沒寫 = 只有選「iAI 聲音」才用）
      "iai": {"voices": ["9D", "36D"], "extra": {...}}   開始畫面列出哪些 iAI 聲音（預設 9D、36D）、額外參數
    """
    try:
        m = os.path.getmtime(TTS_CONF)
    except OSError:
        _conf.update(mtime=None, order=["local"], iai={})
        return _conf
    if m != _conf["mtime"]:
        try:
            import json
            with open(TTS_CONF, encoding="utf-8") as f:
                d = json.load(f)
            order = [x for x in d.get("order", []) if x in ("iai", "local")]
            iai = d.get("iai") if isinstance(d.get("iai"), dict) else {}
            _conf.update(mtime=m, order=order or ["local"], iai=iai)
            logger.info(f"gftts 語音設定：順序 {_conf['order']}，iAI {iai}")
        except Exception as e:
            logger.warning(f"gftts 讀不懂 {TTS_CONF}：{e}")
            _conf.update(mtime=m, order=["local"], iai={})
    return _conf


def _order() -> list:
    return _load_conf()["order"]


def iai_voices() -> list:
    """iAI 的固定聲音清單：data/tts.json 的 iai.voices，沒寫就用 IAI_TTS_VOICES（預設 9D、36D）"""
    c = _load_conf()["iai"]
    raw = c.get("voices")
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list) or not raw:
        raw = IAI_TTS_VOICES.split(",")
    out = []
    for v in raw:
        v = str(v).strip()
        if IAI_NAME_RE.match(v) and v not in out:
            out.append(v)
    return out or ["default"]


def iai_voice_of(choice: str) -> str | None:
    """開始畫面的選擇 → iAI 的 voice：iai_9D → 9D；舊的 iai → 清單第一個；不是 iAI → None"""
    choice = str(choice or "")
    if choice == "iai":
        return iai_voices()[0]
    if choice.startswith("iai_") and choice[4:] in iai_voices():
        return choice[4:]
    return None


def iai_body(text: str, fmt: str = "wav", voice: str | None = None) -> dict:
    """送給 iAI /audio/speech 的內容；voice 用開始畫面選的那個（固定說話者），tts.json 的 iai.extra 會加上去"""
    c = _load_conf()["iai"]
    voice = voice or str(c.get("voice") or "") or iai_voices()[0]
    body = {"model": IAI_TTS_MODEL, "input": text, "voice": voice, "response_format": fmt}
    extra = c.get("extra")
    if isinstance(extra, dict):
        body.update({k: v for k, v in extra.items() if k not in ("model", "input")})
    return body


def _decode(data: bytes):
    """wav 用 soundfile；mp3 等其他格式交給 PyAV（aiortc 自帶）"""
    try:
        stream, sr = sf.read(BytesIO(data), dtype="float32")
        return (stream[:, 0] if stream.ndim > 1 else stream), sr
    except Exception:
        import av
        chunks, sr = [], 24000
        with av.open(BytesIO(data)) as c:
            st = c.streams.audio[0]
            sr = st.rate or 24000
            rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
            for frame in c.decode(st):
                for f in rs.resample(frame):
                    chunks.append(f.to_ndarray().reshape(-1))
            for f in rs.resample(None):
                chunks.append(f.to_ndarray().reshape(-1))
        return (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), sr


@register("tts", "gftts")
class GfTTS(BaseTTS):
    def _fetch_iai(self, text: str, voice: str | None = None) -> bytes | None:
        key = os.getenv("IAI_API_KEY", "")
        if not key:
            logger.warning("iAI 語音：沒有 IAI_API_KEY，跳過")
            return None
        try:
            r = requests.post(IAI_TTS_URL, timeout=IAI_TTS_TIMEOUT,
                              headers={"x-litellm-api-key": f"Bearer {key}", "accept": "application/json"},
                              json=iai_body(text, voice=voice))
            r.raise_for_status()
            if len(r.content) < 200:
                raise ValueError(f"回傳太短：{r.content[:100]!r}")
            return r.content
        except Exception as e:
            logger.warning(f"iAI 語音失敗，改用本機語音：{e}")
            return None

    def _fetch(self, text: str, emotion: str) -> bytes | None:
        order = _order()
        iai_voice = iai_voice_of(getattr(self.parent, "_gf_voice", ""))
        if iai_voice:                                            # 開始畫面選了 iAI 的某個聲音
            order = ["iai", "local"]
        for eng in order:
            if eng == "iai":
                data = self._fetch_iai(text, iai_voice)
                if data:
                    return data
            elif eng == "local":
                data = self._fetch_local(text, emotion)
                if data:
                    return data
        return None

    def _fetch_local(self, text: str, emotion: str) -> bytes | None:
        voice = getattr(self.parent, "_gf_voice", "female")
        if str(voice).startswith("iai"):
            voice = "female"
        for tw in TW_TTS_URLS:
            try:
                r = requests.post(tw, json={"input": text, "voice": voice, "emotion": emotion}, timeout=TIMEOUT)
                r.raise_for_status()
                return r.content
            except Exception as e:
                logger.warning(f"台灣華語語音 {tw} 失敗，試下一個：{e}")
        url = TTS2_URL
        if getattr(self.opt, "TTS_SERVER", ""):
            url = self.opt.TTS_SERVER.rstrip("/") + "/v1/audio/speech"
        try:
            r = requests.post(url, json={"input": text, "emotion": emotion, "response_format": "wav"}, timeout=TIMEOUT)
            r.raise_for_status()
            return r.content
        except Exception as e:
            logger.warning(f"情緒 TTS 失敗，改用 Breeze2-VITS：{e}")
        try:
            r = requests.post(TTS_URL, json={"input": text, "response_format": "wav"}, timeout=TIMEOUT)
            r.raise_for_status()
            return r.content
        except Exception as e:
            logger.error(f"TTS 失敗：{e}")
            return None

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        emotion = (textevent.get("tts") or {}).get("emotion", "neutral")
        t = time.time()
        wav = self._fetch(text, emotion)
        logger.info(f"gftts [{emotion}] {time.time() - t:.2f}s：{text}")
        if not wav:
            return

        try:
            stream, sr = _decode(wav)
        except Exception as e:
            logger.error(f"gftts 解不開語音：{e}")
            return
        if sr != self.sample_rate and stream.shape[0] > 0:
            stream = resampy.resample(x=stream, sr_orig=sr, sr_new=self.sample_rate)
        stream = stream.astype(np.float32)

        # 切成 20ms 一塊交給對嘴；第一塊標 start、最後一塊標 end（網頁字幕靠這兩個事件）
        streamlen, idx = stream.shape[0], 0
        while streamlen >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            streamlen -= self.chunk
            if idx == 0:
                eventpoint = {"status": "start", "text": text}
            elif streamlen < self.chunk:
                eventpoint = {"status": "end", "text": text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
            idx += self.chunk
