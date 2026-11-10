#!/usr/bin/env bash
# 建立並啟動 IndexTTS-2.5 服務（8007），跟 BreezyVoice 共用 data/voices。在 ~/livetalking 底下執行：bash tools/indextts/run.sh
set -e
LT=$(cd "$(dirname "$0")/../.." && pwd)
IDX=${IDX:-$HOME/index-tts}
MODELS=${MODELS:-$HOME/xiaozhi-speech/models}
[ -d "$IDX/indextts" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/index-tts/index-tts.git "$IDX"
if [ ! -f "$MODELS/indextts25/config.yaml" ]; then
  echo "下載 IndexTTS-2.5 模型到 $MODELS/indextts25（第一次，約幾 GB）…"
  docker run --rm --entrypoint python -v "$MODELS":/models xiaozhi-tts2 \
    -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2.5', local_dir='/models/indextts25')"
fi
docker build -t xiaoqing-indextts "$LT/tools/indextts"
docker rm -f xiaoqing-indextts 2>/dev/null || true
docker run -d --name xiaoqing-indextts --gpus all --ipc host --network host --restart unless-stopped \
  -e PORT=8007 -e HF_HOME=/models/hub \
  -v "$IDX":/idx \
  -v "$LT/tools/indextts":/srv:ro \
  -v "$LT/data/voices":/voices:ro \
  -v "$MODELS":/models \
  xiaoqing-indextts
echo "啟動中，看進度：docker logs -f xiaoqing-indextts"
