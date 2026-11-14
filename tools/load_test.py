"""壓力測試：模擬 N 個人同時連線、其中 K 個人同時讓小晴說話，量出「開口要等多久」和「畫面有沒有變卡」。
不用準備 N 台裝置。

在 xiaoqing 容器裡跑（用 127.0.0.1 直連，不經過公開網址、不用 TURN）：
  docker exec xiaoqing python /app/tools/load_test.py --n 15 --speak 5
  docker exec xiaoqing python /app/tools/load_test.py --n 15 --speak 15 --rounds 2
參數：
  --n       同時連線幾人（要 ≤ config.yaml 的 max_session）
  --speak   每一輪同時說話幾人
  --rounds  幾輪
  --voice   用哪個聲音（female／male／iai_9D…）
  --avatar  人物（預設內建小晴；可填 custom_xxx）
  --text    固定說這句（預設三句 24 字左右的長句；聊天時第一小段通常只有 8～12 字）
量什麼：
  開口延遲：送出文字 → 伺服器開始說話（含 TTS 排隊、對嘴）的秒數
  影像流量：每條連線每秒收到多少 KB（掉很多 = 伺服器送不出畫面）
  伺服器每秒格數另外看：docker logs xiaoqing --since 3m 2>&1 | grep "actual avg final fps"
"""
import argparse
import asyncio
import statistics
import time

import aiohttp

SENTENCES = ["大家好，我是小晴，很高興認識你，今天一起加油吧。",
             "這題可以先把大問題拆成三個小步驟，我們一步一步來。",
             "讀了這麼久，要不要先起來喝口水、伸個懶腰再繼續？"]


class Client:
    def __init__(self, base, idx, voice, avatar):
        self.base, self.idx, self.voice, self.avatar = base, idx, voice, avatar
        self.pc = None
        self.sid = None
        self.track_tasks = []
        self.frames = 0                          # 收到（解出）的影像格數 → 算每條連線的 fps

    async def connect(self, http):
        from aiortc import RTCPeerConnection, RTCSessionDescription
        self.pc = RTCPeerConnection()
        self.pc.addTransceiver("video", direction="recvonly")
        self.pc.addTransceiver("audio", direction="recvonly")

        @self.pc.on("track")
        def on_track(track):                     # 一定要把封包收走，不然伺服器那邊會塞住；不解碼，省 CPU
            async def drain():
                try:
                    while True:
                        await track.recv()
                        if track.kind == "video":
                            self.frames += 1
                except Exception:
                    pass
            self.track_tasks.append(asyncio.ensure_future(drain()))

        await self.pc.setLocalDescription(await self.pc.createOffer())
        body = {"sdp": self.pc.localDescription.sdp, "type": "offer", "voice": self.voice}
        if self.avatar:
            body["avatar"] = self.avatar
        r = await http.post(self.base + "/offer", json=body)
        a = await r.json(content_type=None)
        if "sdp" not in a:
            raise RuntimeError(a.get("msg") or repr(a))
        self.sid = a["sessionid"]
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=a["sdp"], type=a["type"]))

    async def bytes_video(self):
        return self.frames                       # （名字沿用）現在回傳收到的影像格數

    async def speaking(self, http):
        r = await http.post(self.base + "/is_speaking", json={"sessionid": self.sid})
        return bool((await r.json(content_type=None)).get("data"))

    async def say(self, http, text):
        """送出 → 等到開始說話，回傳秒數（逾時回傳 None）"""
        t0 = time.time()
        await http.post(self.base + "/human", json={"sessionid": self.sid, "type": "echo", "text": text, "interrupt": True})
        while time.time() - t0 < 40:
            if await self.speaking(http):
                return time.time() - t0
            await asyncio.sleep(0.1)
        return None

    async def wait_quiet(self, http, limit=40):
        t0 = time.time()
        while time.time() - t0 < limit and await self.speaking(http):
            await asyncio.sleep(0.3)

    async def close(self, http):
        try:
            if self.sid:
                await http.post(self.base + "/bye", json={"sessionid": self.sid})
            await self.pc.close()
        except Exception:
            pass
        for t in self.track_tasks:
            t.cancel()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--speak", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--voice", default="female")
    ap.add_argument("--avatar", default="")
    ap.add_argument("--text", default="", help="每輪都說這句（例：--text 嗨，今天過得好嗎 模擬聊天時第一小段）")
    a = ap.parse_args()
    a.speak = min(a.speak, a.n)
    clients = [Client(a.url.rstrip("/"), i, a.voice, a.avatar) for i in range(a.n)]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as http:
        print(f"1. 建立 {a.n} 條連線…", flush=True)
        ok = []
        for c in clients:
            for attempt in range(3):                 # 上一輪測試的連線可能還在釋放，等一下再試
                try:
                    await c.connect(http)
                    ok.append(c)
                    break
                except Exception as e:
                    err = repr(e)
                    try:
                        await c.pc.close()
                    except Exception:
                        pass
                    await asyncio.sleep(5)
            else:
                print(f"   第 {c.idx + 1} 條連不上：{err}")
                break
        print(f"   成功 {len(ok)} 條")
        if not ok:
            return
        await asyncio.sleep(5)

        b0 = [await c.bytes_video() for c in ok]
        await asyncio.sleep(5)
        b1 = [await c.bytes_video() for c in ok]
        idle = [(y - x) / 5 for x, y in zip(b0, b1)]
        print(f"2. 待機時每條連線收到：平均 {statistics.mean(idle):.1f} fps（最低 {min(idle):.1f}）", flush=True)

        lat_all = []
        for rd in range(a.rounds):
            group = ok[(rd * a.speak) % len(ok):][:a.speak]
            if len(group) < a.speak:
                group += ok[:a.speak - len(group)]
            text = a.text or SENTENCES[rd % len(SENTENCES)]
            b0 = [await c.bytes_video() for c in ok]
            t0 = time.time()
            lats = await asyncio.gather(*[c.say(http, text) for c in group])
            await asyncio.gather(*[c.wait_quiet(http) for c in group])
            dt = time.time() - t0
            b1 = [await c.bytes_video() for c in ok]
            kbps = [(y - x) / dt for x, y in zip(b0, b1)]
            got = [x for x in lats if x is not None]
            lat_all += got
            miss = len(lats) - len(got)
            print(f"3.{rd + 1} 同時 {len(group)} 人說話：開口延遲 中位數 {statistics.median(got):.1f}s、最慢 {max(got):.1f}s"
                  + (f"、{miss} 人 40 秒內沒開口" if miss else "")
                  + f"｜這段時間每條連線收到 平均 {statistics.mean(kbps):.1f} fps（最低 {min(kbps):.1f}）", flush=True)
            await asyncio.sleep(2)

        if lat_all:
            print(f"\n總結：{len(ok)} 人在線、每次 {a.speak} 人同時說話 → 開口延遲中位數 {statistics.median(lat_all):.1f}s、最慢 {max(lat_all):.1f}s")
        print("伺服器每秒格數：docker logs xiaoqing --since 3m 2>&1 | grep 'actual avg final fps' | tail -20")
        print("4. 結束連線…")
        await asyncio.gather(*[c.close(http) for c in ok])


if __name__ == "__main__":
    asyncio.run(main())
