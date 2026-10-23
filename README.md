# 小晴 · LiveTalking 即時串流版


以開源專案 LiveTalking（https://github.com/lipku/LiveTalking ，Apache-2.0）為基礎，
在 DGX Spark 上做成虛擬女友「小晴」：WebRTC 即時影像、可打斷、語音輸入、情緒語音、公開網址加通關碼。

> 小晴是 AI 生成的虛擬角色（20 歲成人、日常服裝），不是真人。網頁上一直標示「AI 角色」。
> LiveTalking 的授權要求影片保留 LiveTalking 標示，畫面上已加。

---

## 這個 repo 的內容

- LiveTalking 主程式＋小晴的修改、小晴人物資料（`data/avatars/xiaoqing`）
- **不含**模型權重（GitHub 單檔上限 100 MB）：
  - `models/wav2lip.pth`：LiveTalking README 提供的 wav2lip256.pth，下載後改名放到 `models/`
  - `avatars/wav2lip/face_detection/detection/sfd/s3fd.pth`：第一次產生人物時會自動下載
- **不含**任何金鑰、通關碼、執行紀錄（`.env`、`*.log` 都在 `.gitignore`）
- 小晴用到的語音服務（Whisper、CosyVoice3、Breeze2-VITS）在另一個資料夾 `xiaozhi-speech`，需要時另外提供

---

## 架構

```
瀏覽器 xiaoqing.html
   │  HTTP：/offer、/human、/asr、/sse、/interrupt_talk
   │  WebRTC（UDP）：影像＋聲音
   ▼
LiveTalking（port 8010）
   ├─ llm.py        → 學校 iAI（Nkust 模型）：人設、4 輪記憶、斷句、情緒標籤
   ├─ tts/gftts.py  → tts2 CosyVoice3（8005）帶情緒；失敗改用 tts Breeze2-VITS（8002）
   ├─ wav2lip       → 依聲音逐格畫嘴，貼回 200 格全身畫面，每秒 25 格
   ├─ aiortc        → WebRTC 送出（碼率已調高）
   └─ server/gf_routes.py → /asr 轉給 Whisper（8001）、/gate 通關碼、/iceservers、TURN
cloudflared（lt-tunnel 容器）→ 公開 https 網址（需輸入通關碼）
```

---

## 我新增／修改的檔案（在 livetalking 資料夾內）

| 檔案 | 說明 |
|---|---|
| `Dockerfile.spark`（新） | DGX Spark（ARM）用；以小晴原本的對嘴映像 `xiaozhi-lipsync` 為底 |
| `requirements.txt` | 註解掉 opencv-python-headless（避免版本衝突） |
| `data/avatars/xiaoqing/`（新） | 小晴 200 格全身畫面（含 LiveTalking 標示）、臉部畫面、臉框 coords.pkl |
| `llm.py`（改，原檔 `llm.py.orig`） | 新增 iai provider、小晴人設、記憶、台灣時間、斷句、情緒 |
| `tts/gftts.py`（新） | 情緒語音外掛 |
| `avatars/base_avatar.py`（加 1 行，原檔 `.orig`） | 註冊 gftts |
| `server/gf_routes.py`（新） | 語音輸入、通關碼門禁、STUN/TURN、提高影像碼率 |
| `server/routes.py`（加 1 行，原檔 `.orig`） | 載入 gf_routes |
| `web/xiaoqing.html`（新） | 小晴專用網頁：全螢幕、字幕、可打斷、按住說話、buffer、畫質統計 |

用 `diff llm.py.orig llm.py` 等指令可以看到每個檔案改了什麼。

---

## 執行方式（DGX Spark）

### 1. 先啟動語音服務
```
cd ~/xiaozhi-speech
docker compose up -d asr tts tts2
```
（模型檔第一次啟動時會自動下載到 `models/`，需要 Hugging Face 權限的模型要先設定 token。）

### 2. 建立 LiveTalking 映像（只要一次）
```
cd ~/livetalking
docker build -f Dockerfile.spark -t livetalking-spark .
```
需要先有 `xiaozhi-lipsync` 映像（在 xiaozhi-speech 裡 `docker compose build lipsync`）。

### 3. 啟動小晴
```
cd ~/livetalking
docker run --rm -it --gpus all --ipc host --network host \
  --env-file ~/xiaozhi-speech/.env \
  -v ~/livetalking:/app livetalking-spark \
  python app.py --transport webrtc --model wav2lip --avatar_id xiaoqing \
  --tts gftts --llm_provider iai
```

### 4. 開網頁
- 校內 / VPN：`http://<DGX IP>:8010/xiaoqing.html?mode=chat&buffer=300`
- 加 `&stats=1` 顯示畫質統計
- 麥克風需要 https 或 localhost：校內可用 `ssh -L 18010:localhost:8010 <user>@<DGX IP>` 後開 `http://localhost:18010/...`

### 5.（選用）公開網址
```
docker run -d --name lt-tunnel --network host --restart unless-stopped \
  cloudflare/cloudflared:latest tunnel --no-autoupdate --protocol http2 --url http://localhost:8010
docker logs lt-tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com'
```
從公開網址進來要先輸入通關碼。影像走 UDP，校外沒有 VPN 時需要 TURN（Cloudflare TURN 要綁卡，目前未啟用）。

---

## `.env` 需要的設定（自行建立，`chmod 600`）

| 名稱 | 用途 |
|---|---|
| `IAI_API_KEY` | 學校 iAI 平台金鑰 |
| `GF_ACCESS_CODE` | 公開網址的通關碼 |
| `CF_TURN_KEY_ID`、`CF_TURN_API_TOKEN` | （選用）Cloudflare TURN |
| `GF_VIDEO_KBPS` 等 | （選用）影像碼率，預設起始 4000 kbps；設 0 還原 aiortc 預設 |

---

## 目前狀態

- 完成：人物、iAI 對話、CosyVoice3 情緒語音、自訂網頁、可打斷、公開網址＋通關碼、提高碼率、播放緩衝
- 語音輸入：已上線，反應時間待量測
- 測試：buffer 300 ms 時掉包 748 → 354、卡頓 5 → 3 次、實際碼率 1.6 → 2.8 Mbps（各測一次，初步結果）
- 待做：buffer 設為預設、補對照組測試、換 MuseTalk 讓嘴型更清楚、移植舊版互動功能
- 舊版（一句一段 MP4、MuseTalk）完整保留在 xiaozhi-speech 的 gf / lipsync 服務，可當備用

## 授權與注意

- LiveTalking：Apache-2.0，發布的影片需保留 LiveTalking 標示
- wav2lip 權重、CosyVoice、Whisper 依各自授權；FLUX.1-dev、insightface 為非商用授權，本專案僅供學校展示
- `data/avatars/wav2lip256_avatar1`（LiveTalking 官方範例人物）未打包

---

本專案以 [LiveTalking](https://github.com/lipku/LiveTalking)（Apache-2.0）為基礎修改，原專案說明見 [README-LiveTalking.md](README-LiveTalking.md)。
