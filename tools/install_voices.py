"""把挑好的 Common Voice 候選裝成開始畫面的聲音（女聲 f2…、男聲 m2…）。在主機上、~/livetalking 底下執行：

  python3 tools/install_voices.py --gender female --src ~/xiaozhi-speech/models/cv_pick/female2 --pick 2,5,7,9 --names 柔柔,小安,晴晴,芯芯
  python3 tools/install_voices.py --gender male   --src ~/xiaozhi-speech/models/cv_pick/male2   --pick 1,3,4,6

--pick 是候選編號（cand_2 → 2），--names 可省略（預設「女聲 2」…）。
原本的 female、male 會補上名字「女聲 1」「男聲 1」（已經有 meta.json 就不動）。
裝好不用重開：開始畫面重新整理就看得到，按 ▶ 可以試聽。
"""
import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--gender", required=True, choices=["female", "male"])
ap.add_argument("--src", required=True)
ap.add_argument("--pick", required=True)
ap.add_argument("--names", default="")
ap.add_argument("--voices", default=os.path.join("data", "voices"))
a = ap.parse_args()

src = os.path.expanduser(a.src)
picks = [int(x) for x in a.pick.split(",") if x.strip()]
names = [x.strip()[:12] for x in a.names.split(",")] if a.names else []
zh = "女聲" if a.gender == "female" else "男聲"
prefix = "f" if a.gender == "female" else "m"
os.makedirs(a.voices, exist_ok=True)

base = os.path.join(a.voices, a.gender)
if os.path.isfile(os.path.join(base, "neutral.wav")) and not os.path.exists(os.path.join(base, "meta.json")):
    with open(os.path.join(base, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"name": zh + " 1", "gender": a.gender, "order": 1}, f, ensure_ascii=False)
    print(f"{a.gender} → {zh} 1")

n = 2
for i, k in enumerate(picks):
    while os.path.exists(os.path.join(a.voices, f"{prefix}{n}")):
        n += 1
    vid = f"{prefix}{n}"
    d = os.path.join(a.voices, vid)
    os.makedirs(d)
    shutil.copy(os.path.join(src, f"cand_{k}.wav"), os.path.join(d, "neutral.wav"))
    shutil.copy(os.path.join(src, f"cand_{k}.txt"), os.path.join(d, "neutral.txt"))
    name = names[i] if i < len(names) and names[i] else f"{zh} {n}"
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "gender": a.gender, "order": n, "source": "Mozilla Common Voice zh-TW (CC0)"}, f, ensure_ascii=False)
    print(f"cand_{k} → {vid}（{name}）")
