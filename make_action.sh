#!/bin/bash
# 把 ComfyUI 生成的影片轉成小晴用的動作畫面（25 fps、1280×720、PNG），並修正背景飄移與亮度
# 用法：bash make_action.sh <ComfyUI 輸出的 mp4> <動作名稱，例如 wave_sit> [參考圖]
#   動作會在 poses.json 設 "pingpong": true（播完倒播回來）的話，前面加 PINGPONG=1：
#   PINGPONG=1 bash make_action.sh <mp4> look_window
#   參考圖預設是坐姿待機第 0 格（ComfyUI input 裡的 xiaoqing_sit.png）
# 結果：~/livetalking/data/actions/<動作名稱>/00001.png ...
set -e
SRC="$1"; NAME="$2"
REF="${3:-$HOME/dgx-spark-playbooks/nvidia/playbook-comfyui/input/xiaoqing_sit.png}"
[ -f "$SRC" ] && [ -n "$NAME" ] && [ -f "$REF" ] || { echo "用法：bash make_action.sh <mp4> <名稱> [參考圖]"; exit 1; }
LT=~/livetalking
OUT=$LT/data/actions/$NAME
mkdir -p "$OUT" $LT/data/tmp; rm -f "$OUT"/*.png
cp "$SRC" $LT/data/tmp/_action_src.mp4
cp "$REF" $LT/data/tmp/_action_ref.png

# 1. Wan 2.1 是 16 fps：不管影片檔記錄的速度，一律當 16 fps 讀，再轉成 25 fps、1280×720
docker run --rm -v $LT:/w livetalking-spark ffmpeg -y -loglevel error -i /w/data/tmp/_action_src.mp4 \
  -vf "setpts=N/(16*TB),fps=25,scale=1280:720:flags=lanczos" -start_number 1 /w/data/actions/$NAME/%05d.png

# 2. 修正背景飄移、亮度、顏色，頭尾淡回參考圖
docker run --rm -v $LT:/w -w /w livetalking-spark python3 server/fix_action.py data/actions/$NAME data/tmp/_action_ref.png ${PINGPONG:+--pingpong}

N=$(ls "$OUT"/*.png | wc -l)
echo "完成：$OUT 共 $N 格（約 $((N / 25)) 秒）"
echo "預覽：docker run --rm -v $LT:/w livetalking-spark ffmpeg -y -loglevel error -framerate 25 -i /w/data/actions/$NAME/%05d.png -pix_fmt yuv420p /w/data/tmp/${NAME}_preview.mp4"
