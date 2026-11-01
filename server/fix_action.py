"""修正 Wan 生成的動作影片：背景飄移、亮度忽明忽暗、顏色偏掉

用法（在 livetalking-spark 容器裡跑，make_action.sh 會自動呼叫）：
  python3 fix_action.py <畫面資料夾> <參考圖，例如待機第 0 格> [--keep 0.0]

每一格做四件事：
1. 對齊：用背景區（畫面左側、上方）跟參考圖做相位相關，算出整張畫面的平移，移回去 → 背景不再往右飄
2. 調色：用背景區的 Lab 平均與標準差，把這一格的亮度、色調拉回參考圖 → 不會突然變亮、髮色比較穩
3. 合成：只有「人動到的地方」用生成的畫面，其他地方直接用參考圖 → 背景完全不動
   人動到的範圍 = 所有格跟參考圖差異大的地方的聯集，再擴張、羽化
4. 頭尾各 6 格慢慢淡回參考圖 → 接回待機時不會跳
"""
import glob
import os
import sys

import cv2
import numpy as np

FADE = 6


def bg_mask(h, w):
    """背景參考區：左側 35% 和上方 18%（小晴坐在畫面中間偏右）"""
    m = np.zeros((h, w), np.uint8)
    m[:, : int(w * 0.33)] = 255          # 左邊：植物、牆
    m[:, int(w * 0.80):] = 255           # 右邊：窗戶、沙發
    m[: int(h * 0.15), :] = 255          # 上方
    return m


class Aligner:
    """用背景區的特徵點（ORB）估計「平移＋縮放＋小角度旋轉」，把這一格對回參考圖。
    鏡頭推近、拉遠、平移都能修；特徵點太少時退回只修平移（相位相關）。"""

    def __init__(self, ref, bgm):
        self.orb = cv2.ORB_create(4000)
        self.bgm = bgm
        g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        self.kp_r, self.des_r = self.orb.detectAndCompute(g, bgm)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __call__(self, img):
        h, w = img.shape[:2]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(g, self.bgm)
        if des is None or self.des_r is None or len(kp) < 20:
            return None
        m = sorted(self.bf.match(des, self.des_r), key=lambda x: x.distance)[:600]
        if len(m) < 20:
            return None
        src = np.float32([kp[x.queryIdx].pt for x in m])
        dst = np.float32([self.kp_r[x.trainIdx].pt for x in m])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None or inl is None or inl.sum() < 15:
            return None
        sc = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        if not 0.75 < sc < 1.33 or abs(M[0, 2]) > 250 or abs(M[1, 2]) > 250:
            return None
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out, (float(M[0, 2]), float(M[1, 2]), sc)


def shift_to_ref(img, ref_gray, win, bgm):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = np.where(bgm > 0, g, ref_gray.mean())
    (dx, dy), resp = cv2.phaseCorrelate(ref_gray * win, g * win)
    if resp < 0.05 or abs(dx) > 80 or abs(dy) > 80:      # 算不準就不動
        return img, (0.0, 0.0)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return out, (dx, dy)


def color_to_ref(img, ref_lab_stats, bgm):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = bgm > 0
    out = lab.copy()
    for c in range(3):
        m, s = lab[..., c][sel].mean(), lab[..., c][sel].std() + 1e-3
        rm, rs = ref_lab_stats[c]
        out[..., c] = (lab[..., c] - m) * (rs / s) + rm
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    folder, ref_path = sys.argv[1], sys.argv[2]
    pingpong = "--pingpong" in sys.argv[3:]   # 播完會倒播回來：結尾不要淡回參考圖
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        sys.exit(f"{folder} 裡沒有 png")
    frames = [cv2.imread(f) for f in files]
    h, w = frames[0].shape[:2]
    ref = cv2.resize(cv2.imread(ref_path), (w, h), interpolation=cv2.INTER_AREA)

    bgm = bg_mask(h, w)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_stats = [(ref_lab[..., c][bgm > 0].mean(), ref_lab[..., c][bgm > 0].std()) for c in range(3)]

    # 1+2：對齊、調色
    align = Aligner(ref, bgm) if "--zoom" in sys.argv[3:] else (lambda f: None)   # 預設只修平移（比較穩）
    fixed, shifts, scales = [], [], []
    for f in frames:
        r = align(f)
        if r is not None:
            f, (dx, dy, sc) = r
            d = (dx, dy)
            scales.append(sc)
        else:
            f, d = shift_to_ref(f, ref_gray, win, bgm)
        f = color_to_ref(f, ref_stats, bgm)
        fixed.append(f)
        shifts.append(d)

    # 3：人動到的範圍
    ref_blur = cv2.GaussianBlur(ref, (0, 0), 3).astype(np.int16)
    union = np.zeros((h, w), np.uint8)
    for f in fixed:
        diff = np.abs(cv2.GaussianBlur(f, (0, 0), 3).astype(np.int16) - ref_blur).max(axis=2)
        union |= (diff > 22).astype(np.uint8) * 255
    union = cv2.morphologyEx(union, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    union = cv2.dilate(union, np.ones((41, 41), np.uint8))
    mask = cv2.GaussianBlur(union.astype(np.float32) / 255.0, (0, 0), 15)[..., None]

    # 4：合成＋頭尾淡回參考圖
    n = len(fixed)
    reff = ref.astype(np.float32)
    for i, (path, f) in enumerate(zip(files, fixed)):
        out = f.astype(np.float32) * mask + reff * (1 - mask)
        k = i if pingpong else min(i, n - 1 - i)
        if k < FADE:
            a = (k + 1) / (FADE + 1)
            out = out * a + reff * (1 - a)
        cv2.imwrite(path, np.clip(out, 0, 255).astype(np.uint8))

    sx = [abs(d[0]) for d in shifts]
    zoom = max((abs(1 - v) for v in scales), default=0) * 100
    print(f"修正 {n} 格：最大水平偏移 {max(sx):.1f}px，最大縮放 {zoom:.1f}%，"
          f"特徵點對齊 {len(scales)}/{n} 格，人動到的範圍佔畫面 {mask.mean() * 100:.0f}%")
    cv2.imwrite(os.path.join(os.path.dirname(folder.rstrip('/')), os.path.basename(folder.rstrip('/')) + ".mask.png"),
                (mask[..., 0] * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
