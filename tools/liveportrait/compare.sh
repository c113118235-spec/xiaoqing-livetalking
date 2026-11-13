#!/usr/bin/env bash
# 同一張照片，左邊原本的待機動作（classic）、右邊「AI 補角度」（natural），並排成一支影片比較。
# 在主機上跑：bash ~/livetalking/tools/liveportrait/compare.sh custom_xxxxxxxxxxxxxxxx
# 輸出：~/livetalking/web/_compare.mp4（看完記得刪：rm ~/livetalking/web/_compare.mp4）
set -eu
ID="$1"; ROOT="${LIVETALKING_DIR:-$HOME/livetalking}"
MODELS="${LP_MODELS:-$HOME/xiaozhi-speech/models}"; IMAGE="${LP_IMAGE:-xiaozhi-liveportrait}"
D="$ROOT/data/custom/$ID"; [ -f "$D/canvas.jpg" ] || { echo "找不到 $D/canvas.jpg"; exit 1; }
OUT=$(mktemp -d)
run() { docker run --rm --gpus all --ipc=host -v "$MODELS:/models" -v "$D:/char:ro" -v "$OUT:/out" \
          -v "$ROOT/tools/liveportrait:/gf:ro" "$IMAGE" /gf/idle_gen.py /char/canvas.jpg "/out/$1.mp4" "${@:2}"; }
echo "1/3 原本（classic）…";        run classic --preset classic --seconds 12
echo "2/3 AI 補角度（natural）…";   run natural --preset natural --seconds 12
echo "3/3 並排…"
docker run --rm -v "$OUT:/out" --entrypoint ffmpeg "$IMAGE" -nostdin -loglevel error -y \
  -i /out/classic.mp4 -i /out/natural.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2,scale=1920:-2" \
  -c:v libx264 -crf 20 -pix_fmt yuv420p /out/compare.mp4
cp "$OUT/compare.mp4" "$ROOT/web/_compare.mp4"; rm -rf "$OUT"
echo "完成：瀏覽器開 http://<DGX IP>:8010/_compare.mp4（左：原本，右：AI 補角度）"
