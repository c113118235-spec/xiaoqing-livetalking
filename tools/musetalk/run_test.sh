#!/usr/bin/env bash
# MuseTalk 測試版小晴（port 8011）：原本的 8010（wav2lip）不動，兩邊可以並排比較嘴型。
# 在主機上跑：bash ~/livetalking/tools/musetalk/run_test.sh           # 第一次會先做 MuseTalk 版的人物（幾分鐘）
#            bash ~/livetalking/tools/musetalk/run_test.sh stop      # 關掉測試版
# 看：http://<DGX IP>:8011/xiaoqing.html?mode=chat   （只有區網，沒有接公開網址）
set -eu
LT="${LIVETALKING_DIR:-$HOME/livetalking}"
M="${MUSE_MODELS:-$HOME/xiaozhi-speech/models/musetalk}"
IMAGE="${IMAGE:-livetalking-spark}"
AV="${MUSE_AVATAR:-muse_xiaoqing}"
SRC="${MUSE_SRC:-data/avatars/xiaoqing/full_imgs}"     # 用哪些畫面做人物（資料夾或影片，相對於 livetalking）
NAME=xiaoqing-muse

if [ "${1:-}" = "stop" ]; then docker rm -f "$NAME" >/dev/null 2>&1 && echo "已關閉 $NAME"; exit 0; fi
for d in musetalkV15 sd-vae whisper face-parse-bisent; do
  [ -d "$M/$d" ] || { echo "缺少 $M/$d"; exit 1; }
done
MOUNTS=(-v "$LT:/app" -w /app
        -v "$M/musetalkV15:/app/models/musetalkV15:ro" -v "$M/sd-vae:/app/models/sd-vae:ro"
        -v "$M/whisper:/app/models/whisper:ro" -v "$M/face-parse-bisent:/app/models/face-parse-bisent:ro")

if [ ! -f "$LT/data/avatars/$AV/latents.pt" ]; then
  echo "1/2 做 MuseTalk 版人物 $AV（來源 $SRC）…"
  docker run --rm --gpus all --ipc host "${MOUNTS[@]}" "$IMAGE" \
    python -m avatars.musetalk.genavatar --file "$SRC" --avatar_id "$AV" --version v15
fi

echo "2/2 啟動測試版（port 8011）…"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# 金鑰、通關碼等設定直接沿用正在跑的 xiaoqing 容器（不印出來）
docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --env-file <(docker exec xiaoqing env | grep -E '^(IAI_|GF_|CF_|TW_TTS|TTS)') \
  "${MOUNTS[@]}" "$IMAGE" \
  python app.py --transport webrtc --model musetalk --avatar_id "$AV" --tts gftts --llm_provider iai \
  --listenport 8011 --max_session 1 >/dev/null
echo "啟動中，看進度：docker logs -f $NAME"
echo "好了之後開：http://<DGX IP>:8011/xiaoqing.html?mode=chat&stats=1"
