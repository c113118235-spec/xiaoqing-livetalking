#!/bin/bash
# clone 之後跑一次：把 data/actions_mp4 的影片還原成動作影格
# 用法：docker run --rm -v $PWD:/w -w /w livetalking-spark bash tools/unpack_actions.sh
cd "$(dirname "$0")/.."
for f in data/actions_mp4/*.mp4; do
  n=$(basename "$f" .mp4)
  if [ "$n" = look_turn ]; then out=data/lp_out/look_window; else out=data/actions/$n; fi
  mkdir -p "$out"
  ffmpeg -y -loglevel error -i "$f" -start_number 1 "$out/%05d.png" && echo "$n → $out"
done
