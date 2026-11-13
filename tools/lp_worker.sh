#!/usr/bin/env bash
# 自訂學伴的小幫手：在 DGX 主機上跑（不是在容器裡）。
# 小晴的伺服器把整理好的照片放在 data/custom/<id>/canvas.jpg 並放一個 lp_request 標記，
# 這支程式看到就用 xiaozhi-liveportrait 的 idle_gen.py 做 8 秒待機影片 idle_lp.mp4。
#
# 啟動：  nohup ~/livetalking/tools/lp_worker.sh > ~/livetalking/data/custom/lp_worker.log 2>&1 &
# 開機自動啟動：crontab -e 加一行
#   @reboot sleep 60 && nohup /home/nkust/livetalking/tools/lp_worker.sh > /home/nkust/livetalking/data/custom/lp_worker.log 2>&1 &
set -u
ROOT="${LIVETALKING_DIR:-$HOME/livetalking}"
MODELS="${LP_MODELS:-$HOME/xiaozhi-speech/models}"
IMAGE="${LP_IMAGE:-xiaozhi-liveportrait}"
SECONDS_LOOP="${LP_SECONDS:-8}"
LP_ARGS="${LP_ARGS:-}"                                  # 例如 "--yaw 2 --close-lips 1"
GF_LP="$ROOT/tools/liveportrait"                        # repo 裡的小晴版 idle_gen.py（有就優先用）
CUSTOM="$ROOT/data/custom"
mkdir -p "$CUSTOM"
echo "$(date '+%F %T') 小幫手啟動：$CUSTOM，映像 $IMAGE，模型 $MODELS"

while true; do
  touch "$CUSTOM/.lp_worker" 2>/dev/null               # 心跳：讓小晴知道小幫手活著
  for req in "$CUSTOM"/custom_*/lp_request; do
    [ -e "$req" ] || continue
    dir="$(dirname "$req")"; id="$(basename "$dir")"
    case "$id" in custom_[0-9a-f]*) ;; *) continue ;; esac
    mv "$req" "$dir/lp_running" 2>/dev/null || continue
    echo "$(date '+%F %T') 開始 $id"
    script=/app/idle_gen.py; extra=()
    req_args=""
    argf="$dir/lp_args"; [ -f "$argf" ] || argf="$CUSTOM/lp_args"      # 這一張的參數；沒有就用全部共用的 data/custom/lp_args
    [ -f "$argf" ] && req_args="$(tr -cd 'a-z0-9 ._=-' < "$argf")"     # 例如：--preset natural --seconds 12
    if [ -f "$GF_LP/idle_gen.py" ]; then script=/gf/idle_gen.py; extra=(-v "$GF_LP:/gf:ro"); fi
    if docker run --rm --gpus all --ipc=host \
         -v "$MODELS:/models" -v "$dir:/char" "${extra[@]}" \
         "$IMAGE" "$script" /char/canvas.jpg /char/idle_lp.mp4 --seconds "$SECONDS_LOOP" $LP_ARGS $req_args \
         > "$dir/lp.log" 2>&1 & 
       pid=$!
       while kill -0 "$pid" 2>/dev/null; do touch "$CUSTOM/.lp_worker"; sleep 5; done
       wait "$pid"; then
      touch "$dir/lp_done";   echo "$(date '+%F %T') 完成 $id"
    else
      touch "$dir/lp_failed"; echo "$(date '+%F %T') 失敗 $id：$(tail -n 1 "$dir/lp.log")"
    fi
    rm -f "$dir/lp_running"
  done
  sleep 2
done
