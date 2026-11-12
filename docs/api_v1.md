# 小晴外部串接 API v1：在你的網頁裡放會說話的小晴

小晴是 AI 角色，畫面左上角會一直顯示「AI 角色」，請不要遮住或拿掉。

## 一、運作方式

```
你的後端 ──(sk_ 金鑰)──► POST /api/v1/tokens ──► 入場券（2 分鐘內有效、只能用一次）
    │
    ▼ 把入場券交給你的網頁
你的網頁 ── xiaoqing-embed.js ── iframe(/embed) ──► WebRTC 影音（臉、嘴型、聲音）
    │                                  ▲
    └── xq.say("…") / xq.chat("…") ────┘   （或你的後端用 sk_ 直接呼叫 /say）
```

- **sk_ 金鑰只能放在後端。** 從瀏覽器送出的 sk_ 請求（帶 Origin）一律拒絕。
- 網頁拿到的只有入場券，以及連線後的 session_token。session_token 只能控制自己那一條連線，到時間就失效。

## 二、申請金鑰（小晴管理者做）

```bash
docker exec xiaoqing python /app/tools/api_keys.py create --name partner-a \
    --origin https://a.example.com --max-sessions 1 --minutes 30
```

- 金鑰只會印出這一次。
- 其他指令：`list`、`disable`、`enable`、`rotate`（換新金鑰，舊的立刻失效）、`delete`。
- 加上 `--no-chat`：只能照念，不能讓小晴自己回答。
- 加上 `--iai`：允許使用雲端 iAI 聲音（文字會送到學校 iAI 平台）。

## 三、API

所有請求都要帶 `Authorization: Bearer <金鑰或 token>`，內容一律是 JSON。

| 方法 | 路徑 | 誰用 | 說明 |
|---|---|---|---|
| POST | `/api/v1/tokens` | 後端（sk_） | `{voice?, persona?, greet?}` → `{token, expires_in, voice}` |
| GET | `/api/v1/voices` | 任何人 | 可以選的內建聲音 `{voices:[{id,name,gender}]}` |
| GET | `/api/v1/ice` | 網頁（入場券） | WebRTC 用的 STUN/TURN |
| POST | `/api/v1/sessions` | 網頁（入場券） | `{sdp, type}` → `{session_id, session_token, sdp, type, expires_at}` |
| POST | `/api/v1/whep` | WebRTC 程式（入場券） | 標準 WHEP：body 是 offer SDP → 201 answer SDP（見第五節） |
| DELETE | `/api/v1/whep/{id}` | 同上（session_token） | 結束 WHEP 連線 |
| POST | `/api/v1/sessions/{id}/say` | 網頁（session_token）或後端（sk_） | `{text, emotion?, action?, interrupt?}` 或 `{segments:[...]}`：照念（見第四節）；`mode:"chat"` 讓小晴自己回答；一次最多 300 字 |
| GET | `/api/v1/actions` | 任何人 | 可以用的動作、情緒 |
| WS | `/api/v1/sessions/{id}/audio/stream` | 後端（sk_ 標頭）或網頁（?token=session_token） | 串流送已生好的語音，小晴對嘴（見第四之二節） |
| POST | `/api/v1/sessions/{id}/audio` | 後端或網頁 | 整段語音檔，小晴對嘴 |
| POST | `/api/v1/sessions/{id}/interrupt` | 同上 | 打斷她說話 |
| GET | `/api/v1/sessions/{id}` | 同上 | `{alive, speaking, expires_at}` |
| DELETE | `/api/v1/sessions/{id}` | 同上 | 結束連線，釋放 GPU |

錯誤回應的格式是 `{ok:false, error:"說明"}`，狀態碼如下：

| 狀態碼 | 意思 |
|---|---|
| 401 | 金鑰或 token 不對、過期 |
| 403 | 網域沒被允許、sk_ 放在網頁、不能用 chat |
| 409 | 入場券已經用過 |
| 429 | 太頻繁，或同時連線數已滿 |
| 503 | 伺服器滿了 |

## 四、對方自己做錄音、語音辨識、LLM，小晴只負責「說話＋畫面＋動作」

這是最常見的串法。對方的 LLM 回答好之後，**由對方的後端用 sk_ 呼叫 `/say`**，小晴就會用台灣口音念出來、對嘴，並做動作。

```
使用者 ─錄音─► 對方 App/網頁 ─► 對方後端：語音辨識 → LLM → 回答文字（＋情緒、動作）
                                         │
                                         └─ POST /api/v1/sessions/{id}/say（sk_）
                                                         │
對方 App/網頁裡的 /embed（iframe 或 WebView） ◄─ WebRTC ─ 小晴：TTS ＋ 嘴型 ＋ 動作
```

1. 對方後端：`POST /api/v1/tokens` 換入場券，交給前端。
2. 前端：開 `/embed`（網頁用 `xiaoqing-embed.js`，App 用 WebView），收到 `ready` 時把 `session_id` 回報給自己的後端。
3. 每次 LLM 回答完，後端呼叫：

```bash
curl -X POST https://<小晴網址>/api/v1/sessions/<session_id>/say \
  -H "Authorization: Bearer sk_..." -H "Content-Type: application/json" \
  -d '{"text": "你好！今天想聊什麼？", "emotion": "happy", "action": "wave", "interrupt": true}'
```

一個回答裡不同句子要用不同情緒或動作時，改用 segments：

```json
{"segments": [
  {"text": "恭喜你考過了！", "emotion": "happy", "action": "clap"},
  {"text": "這陣子真的辛苦了。", "emotion": "soft"}
], "interrupt": true}
```

- **emotion**：`neutral`、`happy`、`sad`、`angry`、`soft`，也可以寫中文「平常、開心、難過、生氣、溫柔」。只有 DGX 本機的台灣聲音有情緒變化，iAI 聲音沒有。
- **action**：可以用哪些動作，用 `GET /api/v1/actions` 查，例如 `wave`、`clap`、`thumbs_up`。小晴會先做動作，做完再說話。
- **文字裡的標籤**：如果對方的 LLM 直接輸出 `【開心】【揮手】你好！` 這種標籤，也可以整段丟進 `text`，小晴會把標籤拿掉、不念出來，並照標籤做。
- **interrupt: true**：先停掉她正在說的話，例如使用者又講話了。
- **狀態**：`GET /api/v1/sessions/{id}` 的 `speaking` 可以知道她念完沒，用來決定什麼時候再開麥克風。
- **金鑰設定**：建議建立時加 `--no-chat`，只能照念，不會用到小晴自己的 LLM。

## 四之二、對方送「已經生好的語音」，小晴只對嘴（WebSocket）

對方自己做 TTS，把語音即時串流過來；小晴把聲音和嘴型、動作整合好，透過原本那條 WebRTC 送回對方。這時小晴的聲音就是對方送來的聲音，`emotion` 不會改變聲音。

```
對方 TTS ─PCM─► WS /api/v1/sessions/{id}/audio/stream ─► 小晴：對嘴＋動作 ─WebRTC─► 對方畫面
```

**連線與認證**
- 後端：`wss://<小晴網址>/api/v1/sessions/{id}/audio/stream`，帶標頭 `Authorization: Bearer sk_...`
- 瀏覽器（WebSocket 不能帶標頭）：`.../audio/stream?token=<session_token>`。**sk_ 不能放在網址裡**，會被拒絕。

**訊息**

| 方向 | 訊息 | 說明 |
|---|---|---|
| 對方 → 小晴 | `{"type":"start","format":"pcm_s16le","sample_rate":16000,"channels":1,"action":"wave","interrupt":true}` | 一段語音開始；`action`、`interrupt` 可省略 |
| 對方 → 小晴 | 二進位：PCM 音訊 | 任意長度，建議 20～200 ms 一片 |
| 對方 → 小晴 | `{"type":"end"}` | 這段結束 |
| 對方 → 小晴 | `{"type":"interrupt"}` | 停掉正在說的和排隊中的 |
| 對方 → 小晴 | `{"type":"action","action":"clap"}` | 單獨做一個動作 |
| 小晴 → 對方 | `{"type":"started"}`、`{"type":"done","seconds":1.23}`、`{"type":"error","error":"…"}` | 回覆 |

**音訊格式**：最好是 **16 kHz、單聲道、16-bit PCM（pcm_s16le）**，小晴內部就是用 16 kHz 算嘴型，不用轉。也接受 `pcm_f32le`、8000～48000 Hz、雙聲道，伺服器會自己轉，但會多一點時間和接縫。

**限制**：一段最長 60 秒；排隊還沒播的語音最多 90 秒；一則訊息最大 512 KB。

Python 範例（後端）：

```python
import asyncio, json, aiohttp

async def speak(session_id, pcm_chunks):          # pcm_chunks：你的 TTS 一片一片產生的 16k s16le bytes
    url = f"wss://<小晴網址>/api/v1/sessions/{session_id}/audio/stream"
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(url, headers={"Authorization": "Bearer " + XQ_KEY}) as ws:
            await ws.send_json({"type": "start", "format": "pcm_s16le", "sample_rate": 16000, "interrupt": True})
            await ws.receive_json()                           # {"type":"started"}
            async for chunk in pcm_chunks:
                await ws.send_bytes(chunk)
            await ws.send_json({"type": "end"})
            print(await ws.receive_json())                   # {"type":"done","seconds":…}
```

**不想串流時**：整段上傳也可以。`POST /api/v1/sessions/{id}/audio?action=wave&interrupt=1`，body 放 wav／mp3／ogg 檔；raw PCM 加 `&format=pcm_s16le&sample_rate=16000`。

## 五、直接用 WebRTC 串接（不用 iframe／WebView）

對方用自己的 WebRTC 程式收小晴的影音，例如瀏覽器原生 `RTCPeerConnection`、Android／iOS 的 libwebrtc、Flutter `flutter_webrtc`、GStreamer `whepsrc` 等。

```
對方後端 ──sk_──► POST /api/v1/tokens ──► 入場券
對方前端 ──入場券──► GET /api/v1/ice（拿 STUN/TURN）
         ──入場券──► POST /api/v1/sessions 或 POST /api/v1/whep（送 offer，拿 answer）
         ◄══ WebRTC：影像 H.264（或 VP8）、聲音 Opus，只收不送 ══ 小晴
對方後端 ──sk_──► POST /api/v1/sessions/{id}/say（念 LLM 的回答＋情緒＋動作）
```

建立連線時要照下面做：

| 項目 | 設定 |
|---|---|
| Transceiver | 加兩個 `recvonly`：video、audio（小晴只送不收，不用麥克風和相機） |
| ICE | **不支援 trickle ICE**：要等 ICE 收集完成（或最多等 5 秒）才送 offer，offer 裡要已經有 candidates |
| 編碼 | 影像 H.264 優先，其次 VP8；聲音 Opus |
| STUN/TURN | 用 `GET /api/v1/ice`（帶入場券）拿到的 `iceServers`，在學校網路、手機網路才連得上 |
| 結束 | `DELETE /api/v1/sessions/{id}` 或 `DELETE /api/v1/whep/{id}`，不要只關 PeerConnection |

### 方式 A：JSON（`POST /api/v1/sessions`）

```js
const H = { Authorization: "Bearer " + token };
const { iceServers } = await (await fetch(XQ + "/api/v1/ice", { headers: H })).json();
const pc = new RTCPeerConnection({ iceServers });
pc.addTransceiver("video", { direction: "recvonly" });
pc.addTransceiver("audio", { direction: "recvonly" });
pc.ontrack = (e) => {                                   // 影像、聲音
  if (e.track.kind === "video") videoEl.srcObject = e.streams[0];
  else audioEl.srcObject = e.streams[0];
};
await pc.setLocalDescription(await pc.createOffer());
await new Promise((ok) => {                             // 等 ICE 收集完（不支援 trickle）
  if (pc.iceGatheringState === "complete") return ok();
  pc.onicegatheringstatechange = () => pc.iceGatheringState === "complete" && ok();
  setTimeout(ok, 5000);
});
const r = await fetch(XQ + "/api/v1/sessions", {
  method: "POST", headers: { ...H, "Content-Type": "application/json" },
  body: JSON.stringify({ sdp: pc.localDescription.sdp, type: "offer" }),
});
const s = await r.json();            // {session_id, session_token, sdp, type, expires_at}
await pc.setRemoteDescription({ type: "answer", sdp: s.sdp });
// 把 s.session_id 告訴自己的後端，後端之後用 sk_ 呼叫 /say
```

### 方式 B：標準 WHEP（`POST /api/v1/whep`）

支援 WHEP 的播放器或函式庫可以直接用：

- 請求：`POST /api/v1/whep`，帶 `Content-Type: application/sdp`、`Authorization: Bearer <入場券>`，body 是 offer SDP。
- 回應：`201 Created`，body 是 answer SDP。
- 回應標頭：
  - `Location: /api/v1/whep/{id}`
  - `X-Session-ID`：之後 `/say` 要用的 id
  - `X-Session-Token`：前端控制、結束用
- 結束：`DELETE /api/v1/whep/{id}`，帶 `Authorization: Bearer <X-Session-Token>`。
- 一樣不支援 trickle ICE（PATCH），offer 裡要先有 candidates。

### 自己做畫面時要記得

- 畫面上要顯示「AI 角色」標示，不能拿掉或遮住（用 `/embed` 時會自動顯示）。
- 瀏覽器自動播放聲音需要使用者先點過頁面，所以請在使用者按下按鈕後才建立連線。
- 小晴不說話時也會一直送畫面（待機動作），不用另外處理。

## 六、最簡單的串法

### 後端：換入場券

Node（Express）：

```js
app.post("/xq-token", async (req, res) => {
  const r = await fetch("https://<小晴網址>/api/v1/tokens", {
    method: "POST",
    headers: { Authorization: "Bearer " + process.env.XQ_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ voice: "female" }),
  });
  res.status(r.status).json(await r.json());
});
```

Python（Flask）：

```python
@app.post("/xq-token")
def xq_token():
    r = requests.post("https://<小晴網址>/api/v1/tokens",
                      headers={"Authorization": "Bearer " + os.environ["XQ_KEY"]}, json={"voice": "female"}, timeout=10)
    return r.json(), r.status_code
```

### 網頁

```html
<div id="xq" style="width:360px;height:480px"></div>
<button id="start">和小晴說話</button>
<script src="https://<小晴網址>/xiaoqing-embed.js"></script>
<script>
document.getElementById("start").onclick = async () => {        // 使用者點了才開始，瀏覽器才肯播聲音
  const { token } = await (await fetch("/xq-token", { method: "POST" })).json();
  const xq = XiaoqingEmbed.mount("#xq", { server: "https://<小晴網址>", token });
  xq.on("ready", () => xq.say("哈囉，我是小晴，有什麼想問的嗎？"));
  xq.on("speaking", (v) => console.log("小晴在說話：", v));
  xq.on("needs-gesture", () => console.log("瀏覽器擋了自動播放，請使用者點一下小晴"));
  xq.on("error", (m) => console.warn(m));
  window.xq = xq;          // 之後：xq.chat("今天作業有哪些？")、xq.interrupt()、xq.end()
};
</script>
```

`xiaoqing-embed.js` 的事件：`ready`、`playing`、`speaking`、`needs-gesture`、`ended`、`error`。

## 七、App 用 WebView 串接

App 裡的 WebView 直接打開小晴的 `/embed`，不用再包一層網頁：

```
App ──► 你的後端 POST /xq-token（後端用 sk_ 換入場券）──► App 拿到 token
App 的 WebView 載入  https://<小晴網址>/embed#token=<token>
App 控制：evaluateJavascript("xiaoqing.say('哈囉')")   ← 或你的後端用 sk_ 呼叫 /say
App 收事件：原生橋接（ready / playing / speaking / needs-gesture / ended / error）
```

- sk_ 金鑰一樣只能放在後端，**不能寫進 App**，因為 App 可以被反組譯。
- 指令：`xiaoqing.say(text, interrupt)`、`xiaoqing.chat(text, interrupt)`、`xiaoqing.interrupt()`、`xiaoqing.end()`。
- 事件：小晴頁面會自動找下面任一種橋接，送出 `{xq:"ready"|"speaking"|..., ...}`。

| App 技術 | 橋接名稱 |
|---|---|
| Android 原生 | `addJavascriptInterface(obj, "XiaoqingNative")`，方法 `onEvent(String json)` |
| iOS WKWebView | `userContentController.add(self, name: "xiaoqing")` |
| Flutter webview_flutter | `addJavaScriptChannel("XiaoqingChannel", ...)` |
| Flutter flutter_inappwebview | `addJavaScriptHandler(handlerName: "xiaoqing", ...)` |
| React Native | `onMessage`（`window.ReactNativeWebView.postMessage`） |

### Android（Kotlin）

```kotlin
webView.settings.javaScriptEnabled = true
webView.settings.mediaPlaybackRequiresUserGesture = false      // 讓小晴的聲音可以自動播放
webView.webChromeClient = WebChromeClient()
webView.addJavascriptInterface(object {
    @JavascriptInterface fun onEvent(json: String) {
        val m = JSONObject(json)
        if (m.getString("xq") == "ready") runOnUiThread {
            webView.evaluateJavascript("xiaoqing.say('哈囉，我是小晴！')", null)
        }
    }
}, "XiaoqingNative")
webView.loadUrl("https://<小晴網址>/embed#token=" + token)     // token 從你的後端拿
```

### iOS（Swift）

```swift
let cfg = WKWebViewConfiguration()
cfg.allowsInlineMediaPlayback = true
cfg.mediaTypesRequiringUserActionForPlayback = []
cfg.userContentController.add(self, name: "xiaoqing")          // 收事件
let web = WKWebView(frame: .zero, configuration: cfg)
web.load(URLRequest(url: URL(string: "https://<小晴網址>/embed#token=\(token)")!))
// 讓她說話：web.evaluateJavaScript("xiaoqing.say('哈囉')")
```

### Flutter（webview_flutter）

```dart
final c = WebViewController()
  ..setJavaScriptMode(JavaScriptMode.unrestricted)
  ..addJavaScriptChannel('XiaoqingChannel', onMessageReceived: (m) {
      final e = jsonDecode(m.message);
      if (e['xq'] == 'ready') c.runJavaScript("xiaoqing.say('哈囉，我是小晴！')");
    })
  ..loadRequest(Uri.parse('https://<小晴網址>/embed#token=$token'));
// Android 要讓聲音自動播放：
// (c.platform as AndroidWebViewController).setMediaPlaybackRequiresUserGesture(false);
```

### WebView 注意事項

- 要開 JavaScript，並允許媒體不需點擊就播放（上面的設定）。如果沒開，小晴會出現「▶ 點一下聽小晴說話」按鈕。
- 小晴只會「說」，不會用到 App 的麥克風和相機，所以不用申請這兩個權限。
- App 切到背景或關閉頁面時，請呼叫 `xiaoqing.end()`，把 GPU 讓給別人。
- Android 9 以上預設只允許 HTTPS，小晴網址請一律用 https。

## 八、限制

- **連線數**：DGX 一次只能跑少數幾條影像連線。用完請 `xq.end()`；關掉頁面時也會自動結束。
- **時間**：每條連線最長的分鐘數由金鑰設定，到了自動結束。
- **聲音**：只能用內建聲音。使用者自己錄的聲音和自訂照片學伴，不開放給外部使用。
- **chat 模式**：會用到 LLM，對話內容會經過學校 iAI 平台。
- **嵌入網域**：只有金鑰登記的網域可以把 `/embed` 放進 iframe（由 CSP frame-ancestors 控制）。
- **網址**：正式串接請用固定網址。trycloudflare 的網址每次重開都會變。
