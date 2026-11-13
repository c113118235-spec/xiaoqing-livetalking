# 小晴 AI 學伴 · LiveTalking 即時數位人（DGX Spark）

在 NVIDIA DGX Spark 上即時生成會說話的 AI 角色：打字或用說的，小晴用**台灣口音、帶情緒**的聲音回答，嘴型即時同步，透過 WebRTC 在瀏覽器裡看到。也可以**上傳一張照片**做成自己的學伴，並讀取高科大教學平台的作業與考試放上黑板。

以開源專案 [LiveTalking](https://github.com/lipku/LiveTalking)（Apache-2.0）為基礎修改。

> 小晴是 AI 生成的虛擬角色，不是真人，網頁上一直標示「AI 角色」。
> 照片學伴、聲音克隆都需要**本人同意**；教學平台密碼只用來登入、**不保存、不寫進紀錄**。

---

## 功能

| 類別 | 功能 |
|---|---|
| 人物 | 內建小晴（wav2lip 即時對嘴，25 fps）；動作：揮手、比讚、拍手、看窗外、轉頭（LLM 回答時會自己帶） |
| 照片學伴 | 上傳一張照片 → 去背、素色背景、Real-ESRGAN 放大、LivePortrait 生成待機動作 → 幾分鐘內可以對話；臉框固定大小＋平滑，畫面更穩 |
| 聲音 | IndexTTS-2.5 台灣口音情緒語音（開心、難過、生氣、溫柔、平常）；女聲 5、男聲 5 可試聽；現場錄音克隆自己的聲音；學校 iAI 雲端語音（9D、36D…） |
| 對話 | 學校 iAI 平台 LLM、人設切換（學伴／女友）、可打斷、語音輸入 |
| 學習 | 讀書黑板、提醒；輸入學號密碼由 DGX 登入教學平台，同步這學期的作業與考試 |
| 對外串接 | API v1：金鑰＋一次性入場券、WebRTC（JSON／WHEP）、`/say` 帶情緒與動作、**WebSocket 收對方的 TTS 語音並對嘴**、iframe／App WebView 嵌入（見 [docs/api_v1.md](docs/api_v1.md)） |
| 存取 | 區網直接連；公開網址（Cloudflare Tunnel）＋通關碼；Cloudflare TURN（校外看得到影像） |

---

## 架構

```
瀏覽器 web/xiaoqing.html ─────────────┐        對外串接（docs/api_v1.md）
  HTTP：/offer /human /asr /tts/* …    │          金鑰 → 入場券 → WebRTC
  WebRTC：影像 H.264 ＋ 聲音 Opus      │          /say、WebSocket 語音
                                      ▼
LiveTalking（app.py，port 8010）
 ├─ llm.py              學校 iAI LLM：人設、記憶、斷句、情緒／動作標籤
 ├─ tts/gftts.py        語音：IndexTTS（8007）→ BreezyVoice（8006）→ CosyVoice3（8005）→ Breeze2-VITS（8002）；或學校 iAI 語音
 ├─ wav2lip             依聲音逐格生成嘴型，貼回人物畫面（server/gf_paste.py 柔邊）
 ├─ server/gf_*.py      門禁、TURN、語音輸入、聲音清單、照片學伴、黑板、教學平台同步、姿勢動作、對外 API
 └─ aiortc              WebRTC 送出（碼率已調高、可設定聲音延後對齊嘴型）

主機上的小幫手 tools/lp_worker.sh ──► LivePortrait 容器（照片學伴的待機影片）
cloudflared 容器 ──► 公開 https 網址
```

| 服務 | Port | 位置 |
|---|---|---|
| LiveTalking（小晴） | 8010 | 本 repo，容器 `xiaoqing` |
| IndexTTS-2.5 | 8007 | `tools/indextts/`，容器 `xiaoqing-indextts` |
| BreezyVoice（備援） | 8006 | `tools/breezy/`，容器 `xiaoqing-breezy` |
| CosyVoice3、Breeze2-VITS、Whisper | 8005、8002、8001 | 另一個資料夾 `xiaozhi-speech` |
| LivePortrait | — | 映像 `xiaozhi-liveportrait`，由 `tools/lp_worker.sh` 呼叫 `tools/liveportrait/idle_gen.py` |

---

## 執行方式（DGX Spark）

> **Docker 一鍵部署（docker compose）整理中，11/20 前完成。** 下面是目前的啟動方式。

1. **語音服務**
   ```bash
   cd ~/xiaozhi-speech && docker compose up -d asr tts tts2
   bash ~/livetalking/tools/indextts/run.sh          # IndexTTS-2.5（第一次會下載模型）
   ```
2. **小晴**
   ```bash
   cd ~/livetalking
   docker build -f Dockerfile.spark -t livetalking-spark .      # 只要一次（需要 xiaozhi-lipsync 映像）
   docker run -d --name xiaoqing --gpus all --ipc host --network host --restart unless-stopped \
     --env-file ~/xiaozhi-speech/.env -v ~/livetalking:/app livetalking-spark \
     python app.py --transport webrtc --model wav2lip --avatar_id xiaoqing --tts gftts --llm_provider iai
   ```
3. **照片學伴的小幫手**（主機上）
   ```bash
   nohup ~/livetalking/tools/lp_worker.sh > ~/livetalking/data/custom/lp_worker.log 2>&1 &
   ```
4. **開網頁**：`http://<DGX IP>:8010/xiaoqing.html?mode=chat`（加 `&stats=1` 顯示畫質統計）
   - 麥克風需要 https 或 localhost（可用 `ssh -L 18010:localhost:8010 …` 再開 `http://localhost:18010/…`）
5. **（選用）公開網址**
   ```bash
   docker run -d --name xiaoqing-tunnel --network host --restart unless-stopped \
     cloudflare/cloudflared:latest tunnel --no-autoupdate --protocol http2 --url http://localhost:8010
   docker logs xiaoqing-tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com'
   ```

### 不在 repo 裡的東西

- **模型權重**：`models/wav2lip.pth`（LiveTalking 的 wav2lip256）、IndexTTS-2.5、LivePortrait、Real-ESRGAN 等，各自的下載方式寫在對應的 `run.sh` 或程式開頭
- **聲音**：`data/voices/`（Mozilla Common Voice zh-TW，CC0），用 `tools/indextts/pick_cv_voice.py` 挑、`tools/install_voices.py` 安裝
- **使用者資料**：照片學伴、自己錄的聲音、金鑰（`data/custom/`、`data/voices/u_*`、`data/api/`）都不進 git

---

## 設定

`.env`（自行建立、`chmod 600`、**不要 commit**）：

| 名稱 | 用途 |
|---|---|
| `IAI_API_KEY` | 學校 iAI 平台金鑰（LLM 與 iAI 語音共用） |
| `GF_ACCESS_CODE` | 公開網址的通關碼 |
| `CF_TURN_KEY_ID`、`CF_TURN_API_TOKEN` | （選用）Cloudflare TURN，校外看得到影像 |
| `GF_VIDEO_KBPS`、`GF_VIDEO_MIN_KBPS`、`GF_VIDEO_MAX_KBPS` | （選用）影像碼率，預設 4000／2000／8000 kbps |
| `CUSTOM_STABLE_BOX` | 照片學伴臉框固定大小＋平滑，預設 1 |

改了馬上生效、不用重開的檔案：

| 檔案 | 例子 |
|---|---|
| `data/tts.json` | `{"iai": {"voices": ["9D", "36D"]}}` 列出 iAI 聲音；加 `"order": ["iai","local"]` 則全部先用 iAI |
| `data/av.json` | `{"audio_delay_ms": 120}` 聲音延後，讓嘴型對齊（新連線生效） |
| `data/api/keys.json` | 對外金鑰，用 `tools/api_keys.py` 管理 |

---

## 工具

| 工具 | 用途 |
|---|---|
| `tools/api_keys.py` | 建立／停用／換新對外金鑰 |
| `tools/ws_audio_test.py` | 端對端測試：WebSocket 送語音 → 小晴對嘴 → WebRTC 錄成 mp4 |
| `tools/avatar_stability.py` | 量測、修正人物臉框穩定度（改之前自動備份，可還原） |
| `tools/iai_probe.py` | 測試 iAI 語音的參數 |
| `tools/install_voices.py` | 安裝內建聲音 |
| `tools/lp_worker.sh`、`tools/liveportrait/idle_gen.py` | 照片學伴的待機影片 |

---

## 授權與注意

- LiveTalking：Apache-2.0，影片需保留 LiveTalking 標示（畫面上已加）
- wav2lip、IndexTTS、CosyVoice、Whisper、LivePortrait、Real-ESRGAN 依各自授權；insightface 等為非商用授權，本專案僅供學校研究與展示
- 不可用未經同意的真人照片或聲音（包含藝人、公眾人物）
- 原專案說明見 [README-LiveTalking.md](README-LiveTalking.md)
