#!/usr/bin/env bash
# 一台 DGX 開多個小晴程序（worker）分攤連線。主程序 8010 會依 data/workers.json 把滿出來的連線分給 worker。
# 在主機上跑：
#   bash ~/livetalking/tools/workers.sh start 2      # 開 2 個 worker（port 8021、8022），每個最多 5 人，並寫好 data/workers.json
#   bash ~/livetalking/tools/workers.sh status
#   bash ~/livetalking/tools/workers.sh stop         # 關掉全部 worker，刪掉 data/workers.json（回到只有主程序）
# worker 的映像、掛載、環境變數、啟動參數都照抄正在跑的 xiaoqing 容器，只改 port 和人數上限。
set -eu
LT="${LIVETALKING_DIR:-$HOME/livetalking}"
CMD="${1:-status}"; N="${2:-2}"; PER="${PER:-5}"; LOCAL_MAX="${LOCAL_MAX:-5}"; BASE=8020

case "$CMD" in
stop)
  for c in $(docker ps -a --format '{{.Names}}' | grep -E '^xiaoqing-w[0-9]+$' || true); do docker rm -f "$c" >/dev/null && echo "已關閉 $c"; done
  rm -f "$LT/data/workers.json"; echo "已刪除 data/workers.json（只剩主程序）"; exit 0 ;;
status)
  docker ps --format '{{.Names}}\t{{.Status}}' | grep -E '^xiaoqing(-w[0-9]+)?\b' || true
  [ -f "$LT/data/workers.json" ] && cat "$LT/data/workers.json"
  echo; curl -s -m 5 "http://127.0.0.1:8010/api/admin/sessions?all=1"; echo; exit 0 ;;
start) ;;
*) echo "用法：$0 start N | status | stop"; exit 1 ;;
esac

docker inspect xiaoqing >/dev/null || { echo "找不到 xiaoqing 容器"; exit 1; }
ENVF=$(mktemp); trap 'rm -f "$ENVF"' EXIT
# 照抄 xiaoqing 的設定（用 python 讀 docker inspect，避免手動抄錯）
docker inspect xiaoqing > "$ENVF.json"
python3 - "$ENVF.json" "$ENVF" > "$ENVF.args" <<'PY'
import json, sys, shlex
c = json.load(open(sys.argv[1]))[0]
with open(sys.argv[2], "w") as f:
    for e in c["Config"]["Env"]:
        if not e.startswith(("PATH=", "HOSTNAME=", "HOME=")):
            f.write(e + "\n")
binds = c["HostConfig"].get("Binds") or []
cmd = c["Config"]["Cmd"] or []
print(c["Config"]["Image"])
print(" ".join(shlex.quote(b) for b in binds))
print(" ".join(shlex.quote(x) for x in cmd))
print(c["Config"].get("WorkingDir") or "/app")
PY
rm -f "$ENVF.json"
IMAGE=$(sed -n 1p "$ENVF.args"); BINDS=$(sed -n 2p "$ENVF.args"); APPCMD=$(sed -n 3p "$ENVF.args"); WD=$(sed -n 4p "$ENVF.args")
rm -f "$ENVF.args"
# 啟動參數：去掉原本的 --listenport / --max_session，換成 worker 的
APPCMD=$(python3 - "$APPCMD" <<'PY'
import shlex, sys
a = shlex.split(sys.argv[1]); out = []; skip = False
for x in a:
    if skip: skip = False; continue
    if x in ("--listenport", "--max_session"): skip = True; continue
    if x.startswith(("--listenport=", "--max_session=")): continue
    out.append(x)
print(" ".join(shlex.quote(x) for x in out))
PY
)
WORKERS=""
for i in $(seq 1 "$N"); do
  name="xiaoqing-w$i"; port=$((BASE + i))
  docker rm -f "$name" >/dev/null 2>&1 || true
  eval "docker run -d --name $name --gpus all --ipc host --network host --restart unless-stopped \
    --env-file $ENVF -e GF_WORKER=1 $(for b in $BINDS; do printf -- '-v %s ' "$b"; done) -w $WD \
    $IMAGE $APPCMD --listenport $port --max_session $PER" >/dev/null
  echo "已啟動 $name（port $port，最多 $PER 人）"
  WORKERS="$WORKERS{\"url\": \"http://127.0.0.1:$port\", \"max\": $PER},"
done
echo "{\"local_max\": $LOCAL_MAX, \"workers\": [${WORKERS%,}]}" > "$LT/data/workers.json"
echo "已寫入 data/workers.json：主程序 $LOCAL_MAX 人＋$N 個 worker × $PER 人 = $((LOCAL_MAX + N * PER)) 人"
echo "worker 啟動要 30～60 秒；看狀態：bash $0 status"
