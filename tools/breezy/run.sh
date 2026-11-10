#!/usr/bin/env bash
# 建立並啟動 BreezyVoice 服務（8006）。在 ~/livetalking 底下執行：bash tools/breezy/run.sh
set -e
LT=$(cd "$(dirname "$0")/../.." && pwd)
BV=${BV:-$HOME/breezyvoice}
[ -d "$BV/cosyvoice" ] || git clone https://github.com/mtkresearch/BreezyVoice.git "$BV"
mkdir -p "$LT/data/voices/female" "$LT/data/voices/male"
MODELS=${MODELS:-$HOME/xiaozhi-speech/models}     # 跟 tts2 共用模型資料夾（HF 快取在 /models/hub）

# 文字正規化規則檔：實驗室網路連不到 modelscope.cn，從 tts2 容器複製一份
if [ ! -f "$BV/wetext/zh/tn/tagger.fst" ]; then
  SRC=$(docker exec xiaozhi-tts2 sh -c 'find / -xdev -path "*wetext*" -name tagger.fst 2>/dev/null | grep "/zh/tn/" | head -1' || true)
  if [ -n "$SRC" ]; then
    REPO=${SRC%/zh/tn/tagger.fst}
    mkdir -p "$BV/wetext" && docker cp "xiaozhi-tts2:$REPO/." "$BV/wetext/" && echo "文字正規化規則檔：從 tts2 複製（$REPO）"
  else
    echo "tts2 裡找不到 wetext 規則檔，改用簡易數字轉中文"
  fi
fi
mkdir -p "$BV/wetext"

# 還沒有女聲就先用小晴原本（本人同意）的參考錄音；再沒有就用 BreezyVoice 附的範例
F=$LT/data/voices/female
if [ ! -f "$F/neutral.wav" ]; then
  if [ -f "$HOME/xiaozhi-speech/gf/voice/prompt.wav" ]; then
    cp "$HOME/xiaozhi-speech/gf/voice/prompt.wav" "$F/neutral.wav"
    cp "$HOME/xiaozhi-speech/gf/voice/prompt.txt" "$F/neutral.txt"
    echo "女聲：先用小晴原本的參考錄音"
  else
    cp "$BV/data/example.wav" "$F/neutral.wav"
    echo "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。只有擁有解密方法的對象，經由解密過程，才能將密文還原為正常可讀的內容。" > "$F/neutral.txt"
    echo "女聲：先用 BreezyVoice 的範例錄音"
  fi
fi

docker build -t xiaoqing-breezy "$LT/tools/breezy"
docker rm -f xiaoqing-breezy 2>/dev/null || true
docker run -d --name xiaoqing-breezy --gpus all --ipc host --network host --restart unless-stopped \
  -e PORT=8006 \
  -v "$BV":/bv \
  -v "$LT/tools/breezy":/srv:ro \
  -v "$LT/data/voices":/voices:ro \
  -v "$MODELS":/models \
  -v "$BV/wetext":/wetext:ro \
  xiaoqing-breezy
echo "啟動中，看進度：docker logs -f xiaoqing-breezy"
