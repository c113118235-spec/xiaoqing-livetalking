"""小晴的情緒語音。

- 依序試 TW_TTS_URLS：IndexTTS-2.5（8007，tools/indextts）→ BreezyVoice（8006，tools/breezy）
  兩個都用開始畫面選的聲音（female / male）＋情緒
- 都失敗 → tts2（CosyVoice3，8005）→ 再失敗 → tts（Breeze2-VITS，8002）
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


@register("tts", "gftts")
class GfTTS(BaseTTS):
    def _fetch(self, text: str, emotion: str) -> bytes | None:
        voice = getattr(self.parent, "_gf_voice", "female")
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

        stream, sr = sf.read(BytesIO(wav), dtype="float32")
        if stream.ndim > 1:
            stream = stream[:, 0]
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
