"""端對端測試「對方用 WebSocket 送 TTS 語音 → 小晴對嘴 → WebRTC 回傳」，不用等對方。

在 xiaoqing 容器裡跑（那裡有 aiortc）：
  docker exec -e XQ_KEY=sk_217_... xiaoqing python /app/tools/ws_audio_test.py --wav /app/data/test.wav
它會：
  1. 用金鑰換入場券
  2. 用 aiortc 當「對方」建 WebRTC 連線（只收不送），把收到的影音錄成 data/ws_test.mp4
  3. 把 wav 轉成 16 kHz PCM，用 WebSocket 一片一片送過去（模擬對方的串流 TTS），順便揮手
  4. 印出「送出第一片 → 小晴開始說話」的延遲，錄完後結束連線
看 data/ws_test.mp4 就知道嘴型、聲音、動作對不對。
"""
import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp
import numpy as np


def load_pcm16k(path):
    import soundfile as sf
    try:
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        a = a.mean(axis=1)
    except Exception:
        import av                                   # mp3 等
        chunks, sr = [], 16000
        with av.open(path) as c:
            st = c.streams.audio[0]
            rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
            for fr in c.decode(st):
                for f in rs.resample(fr):
                    chunks.append(f.to_ndarray().reshape(-1))
        a = np.concatenate(chunks)
    if sr != 16000:
        import resampy
        a = resampy.resample(a, sr, 16000)
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


class Recorder:
    """把收到的影音照「原本的解析度」錄成 mp4（aiortc 內建的 MediaRecorder 會固定 640x480，畫面被裁掉）"""

    def __init__(self, path):
        import av
        self.av, self.path = av, path
        self.tracks, self.tasks = [], []
        self.c = self.vs = self.as_ = None
        self.pending = []

    def add(self, track):
        self.tracks.append(track)

    async def start(self):
        self.tasks = [asyncio.ensure_future(self._run(t)) for t in self.tracks]

    def _open(self, vframe):
        av = self.av
        self.c = av.open(self.path, "w")
        self.vs = self.c.add_stream("libx264", rate=25)
        self.vs.width, self.vs.height, self.vs.pix_fmt = vframe.width // 2 * 2, vframe.height // 2 * 2, "yuv420p"
        self.as_ = self.c.add_stream("aac", rate=48000) if any(t.kind == "audio" for t in self.tracks) else None

    def _write(self, frame, kind):
        st = self.vs if kind == "video" else self.as_
        if st is None:
            return
        for pkt in st.encode(frame):
            self.c.mux(pkt)

    async def _run(self, track):
        while True:
            try:
                f = await track.recv()
            except Exception:
                return
            if self.c is None:
                if track.kind == "video":
                    self._open(f)
                    self._write(f, "video")
                    for k, pf in self.pending:
                        self._write(pf, k)
                    self.pending = []
                else:
                    self.pending.append(("audio", f))
                continue
            self._write(f, track.kind)

    async def stop(self):
        for t in self.tasks:
            t.cancel()
        await asyncio.sleep(0.1)
        if self.c is not None:
            for st in (self.vs, self.as_):
                if st is not None:
                    for pkt in st.encode(None):
                        self.c.mux(pkt)
            self.c.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010", help="小晴伺服器（容器裡用 127.0.0.1）")
    ap.add_argument("--wav", required=True, help="要讓小晴說的語音檔（wav／mp3）")
    ap.add_argument("--action", default="wave", help="開口前的動作，空字串 = 不做")
    ap.add_argument("--chunk-ms", type=int, default=40, help="每片幾毫秒（模擬串流 TTS）")
    ap.add_argument("--out", default="data/ws_test.mp4")
    a = ap.parse_args()
    key = os.getenv("XQ_KEY", "")
    if not key.startswith("sk_"):
        sys.exit("請用 -e XQ_KEY=sk_... 帶金鑰（不要寫在指令參數裡）")
    base = a.url.rstrip("/")
    pcm = load_pcm16k(a.wav)
    secs = len(pcm) / 32000
    print(f"語音 {secs:.1f} 秒")

    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

    async with aiohttp.ClientSession() as http:
        H = {"Authorization": "Bearer " + key}
        r = await http.post(base + "/api/v1/tokens", headers=H, json={})
        tok = (await r.json()).get("token")
        if not tok:
            sys.exit(f"換入場券失敗：{r.status} {await r.text()}")
        print("1. 拿到入場券")

        r = await http.get(base + "/api/v1/ice", headers={"Authorization": "Bearer " + tok})
        ice = [RTCIceServer(**{k: v for k, v in s.items() if k in ("urls", "username", "credential")})
               for s in (await r.json()).get("iceServers", [])[:2]] if r.status == 200 else []
        pc = RTCPeerConnection(RTCConfiguration(iceServers=ice))
        rec = Recorder(a.out)
        pc.on("track", rec.add)
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")
        await pc.setLocalDescription(await pc.createOffer())       # aiortc 會等 ICE 收集完
        r = await http.post(base + "/api/v1/sessions", headers={"Authorization": "Bearer " + tok},
                            json={"sdp": pc.localDescription.sdp, "type": "offer"})
        s = await r.json()
        if r.status != 200:
            sys.exit(f"建立連線失敗：{r.status} {s}")
        sid = s["session_id"]
        await pc.setRemoteDescription(RTCSessionDescription(sdp=s["sdp"], type="answer"))
        await rec.start()
        print(f"2. WebRTC 連上，session_id={sid}，錄影到 {a.out}")
        await asyncio.sleep(3)                                      # 先錄 3 秒待機畫面

        async def speaking():
            r = await http.get(f"{base}/api/v1/sessions/{sid}", headers=H)
            return (await r.json()).get("speaking")

        ws_url = base.replace("http", "ws", 1) + f"/api/v1/sessions/{sid}/audio/stream"
        async with http.ws_connect(ws_url, headers=H) as ws:
            await ws.send_json({"type": "start", "format": "pcm_s16le", "sample_rate": 16000,
                                **({"action": a.action} if a.action else {})})
            print("3.", await ws.receive_json())
            step = 32 * a.chunk_ms                                  # 16k × 2 bytes × ms
            t0, first_talk = time.time(), None
            for i in range(0, len(pcm), step):
                await ws.send_bytes(pcm[i:i + step])
                await asyncio.sleep(a.chunk_ms / 1000)              # 照真實速度送，像串流 TTS
                if first_talk is None and await speaking():
                    first_talk = time.time() - t0
            await ws.send_json({"type": "end"})
            print("  ", await ws.receive_json())
        while first_talk is None and time.time() - t0 < 15:
            if await speaking():
                first_talk = time.time() - t0
            await asyncio.sleep(0.05)
        print(f"4. 送出第一片 → 小晴開始說話：{first_talk:.2f} 秒（含動作時間）" if first_talk else "4. 沒偵測到她開始說話")
        t1 = time.time()
        while await speaking() and time.time() - t1 < secs + 10:
            await asyncio.sleep(0.2)
        await asyncio.sleep(1.5)
        await rec.stop()
        await http.delete(f"{base}/api/v1/sessions/{sid}", headers=H)
        await pc.close()
        print(f"完成。看 {a.out}：嘴型跟聲音有沒有對上、有沒有先揮手。")


if __name__ == "__main__":
    asyncio.run(main())
