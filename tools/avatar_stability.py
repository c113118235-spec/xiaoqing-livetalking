"""量／改善人物的「臉框穩定度」（自訂學伴比內建小晴抖，主要就是臉框每格忽大忽小、跳動）。

在 xiaoqing 容器裡跑：
  # 量：每格臉框中心移動幾像素、大小變幾次
  docker exec xiaoqing python /app/tools/avatar_stability.py measure custom_xxxxxxxxxxxxxxxx
  docker exec xiaoqing python /app/tools/avatar_stability.py measure xiaoqing          # 跟內建小晴比
  # 改：用已經做好的畫面重新找臉，改成固定大小＋平滑的臉框（不用重跑 LivePortrait，幾十秒）
  docker exec xiaoqing python /app/tools/avatar_stability.py fix custom_xxxxxxxxxxxxxxxx
  # 改壞了想還原
  docker exec xiaoqing python /app/tools/avatar_stability.py restore custom_xxxxxxxxxxxxxxxx

fix 之前會把原本的 coords.pkl 和 face_imgs 備份到 <人物>/backup_box/；改完要 docker restart xiaoqing 才會用新的（人物載入後會被快取）。
"""
import argparse
import glob
import os
import pickle
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVATARS = os.path.join(ROOT, "data", "avatars")


def _path(aid):
    if not aid.replace("_", "").isalnum():
        sys.exit("人物 id 只能有英數字和底線")
    p = os.path.join(AVATARS, aid)
    if not os.path.isfile(os.path.join(p, "coords.pkl")):
        sys.exit(f"找不到 {p}/coords.pkl")
    return p


def stats(coords):
    """coords：每格 (y1, y2, x1, x2)"""
    c = np.array(coords, dtype=np.float64)
    cx, cy = (c[:, 2] + c[:, 3]) / 2, (c[:, 0] + c[:, 1]) / 2
    w, h = c[:, 3] - c[:, 2], c[:, 1] - c[:, 0]
    d = np.hypot(np.diff(cx), np.diff(cy))
    dw = np.abs(np.diff(w)) + np.abs(np.diff(h))
    return {"格數": len(c), "中心平均每格移動(px)": round(float(d.mean()), 2), "中心最大一次跳動(px)": round(float(d.max()), 1),
            "框大小改變的格數": int((dw > 0).sum()), "框寬範圍(px)": f"{int(w.min())}–{int(w.max())}"}


def measure(aid):
    with open(os.path.join(_path(aid), "coords.pkl"), "rb") as f:
        s = stats(pickle.load(f))
    print(f"{aid}：")
    for k, v in s.items():
        print(f"  {k}：{v}")
    return s


def fix(aid):
    sys.path.insert(0, ROOT)
    import cv2
    import torch
    from avatars.wav2lip import face_detection
    from avatars.wav2lip.genavatar import stabilize_boxes
    p = _path(aid)
    full = sorted(glob.glob(os.path.join(p, "full_imgs", "*.png")))
    faces = sorted(glob.glob(os.path.join(p, "face_imgs", "*.png")))
    if not full or not faces:
        sys.exit("沒有 full_imgs／face_imgs")
    size = cv2.imread(faces[0]).shape[0]
    print("改之前：")
    before = measure(aid)
    bk = os.path.join(p, "backup_box")
    if not os.path.isdir(bk):                                   # 只備份第一次（原始版）
        os.makedirs(bk)
        shutil.copy2(os.path.join(p, "coords.pkl"), bk)
        shutil.copytree(os.path.join(p, "face_imgs"), os.path.join(bk, "face_imgs"))
    frames = [cv2.imread(f) for f in full]
    det = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                       device="cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    for i in range(0, len(frames), 8):
        preds.extend(det.get_detections_for_batch(np.array(frames[i:i + 8])))
    pady1, pady2, padx1, padx2 = 0, 10, 0, 0                    # 跟 genavatar 預設一樣
    rects = [None if r is None else [max(0, r[0] - padx1), max(0, r[1] - pady1),
                                     min(fr.shape[1], r[2] + padx2), min(fr.shape[0], r[3] + pady2)]
             for r, fr in zip(preds, frames)]
    boxes = stabilize_boxes(rects, frames[0].shape)
    coords = []
    for i, ((x1, y1, x2, y2), fr) in enumerate(zip(boxes, frames)):
        cv2.imwrite(os.path.join(p, "face_imgs", f"{i:08d}.png"), cv2.resize(fr[y1:y2, x1:x2], (size, size)))
        coords.append((int(y1), int(y2), int(x1), int(x2)))
    with open(os.path.join(p, "coords.pkl"), "wb") as f:
        pickle.dump(coords, f)
    print(f"\n改之後（{sum(r is None for r in rects)} 格沒找到臉已內插）：")
    after = measure(aid)
    print(f"\n中心平均每格移動：{before['中心平均每格移動(px)']} → {after['中心平均每格移動(px)']} px；"
          f"框大小改變：{before['框大小改變的格數']} → {after['框大小改變的格數']} 格")
    print("要用新的臉框：docker restart xiaoqing（人物會被快取），再重新整理網頁選這個學伴。")


def restore(aid):
    p = _path(aid)
    bk = os.path.join(p, "backup_box")
    if not os.path.isdir(bk):
        sys.exit("沒有備份")
    shutil.copy2(os.path.join(bk, "coords.pkl"), p)
    shutil.rmtree(os.path.join(p, "face_imgs"))
    shutil.copytree(os.path.join(bk, "face_imgs"), os.path.join(p, "face_imgs"))
    print("已還原原本的臉框")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "fix", "restore"])
    ap.add_argument("avatar_id")
    a = ap.parse_args()
    {"measure": measure, "fix": fix, "restore": restore}[a.cmd](a.avatar_id)
