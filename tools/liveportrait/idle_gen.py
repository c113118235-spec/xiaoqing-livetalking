"""用 LivePortrait 幫角色圖做「待機小動作」循環影片：眨眼、頭微微左右／上下擺、眼神飄一下。

跟 Wan 不同：LivePortrait 只移動臉部的關鍵點，背景、身體、鏡頭完全不動，
而且每個動作的大小、時間點都由這支程式控制，不會突然轉身或換臉。

用法（容器裡）：
  python3 idle_gen.py /char/char.png /char/idle.mp4 [--seconds 8] [--yaw 3] [--pitch 1.5] [--roll 1.5]
頭擺動用正弦波、起點跟終點都是原本的姿勢，所以影片可以無縫循環。

小晴版（tools/liveportrait/idle_gen.py，lp_worker.sh 會優先用這支）多了：
  --close-lips 1   照片裡嘴巴張開或露齒時，整段待機都把嘴巴調成閉合：wav2lip 貼嘴型時比較不會有接縫、閃爍
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

# 舊格式權重（PyTorch 2.6 以後 torch.load 預設 weights_only=True 會讀不進來）
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": k.get("weights_only", False)})

# onnxruntime 在 ARM 上只有 CPU 版：把不存在的 CUDA provider 拿掉（找臉只做一次，CPU 就夠快）
import onnxruntime as ort  # noqa: E402
class _Session(ort.InferenceSession):
    """跟原本的 InferenceSession 一樣（insightface 會繼承它），只是自動拿掉這台機器沒有的 provider"""
    def __init__(self, path, *a, providers=None, **k):
        avail = set(ort.get_available_providers())
        if providers:
            providers = [p for p in providers
                         if (p[0] if isinstance(p, (tuple, list)) else p) in avail] or ["CPUExecutionProvider"]
        super().__init__(path, *a, providers=providers, **k)


ort.InferenceSession = _Session

LP = os.getenv("LIVEPORTRAIT_SRC", "/opt/LivePortrait")
WEIGHTS = os.getenv("LP_WEIGHTS", "/models/liveportrait")
sys.path.insert(0, LP)
from huggingface_hub import snapshot_download  # noqa: E402

from src.config.crop_config import CropConfig  # noqa: E402
from src.config.inference_config import InferenceConfig  # noqa: E402
from src.live_portrait_pipeline import LivePortraitPipeline  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402
from src.utils.crop import paste_back, prepare_paste_back  # noqa: E402
from src.utils.io import load_image_rgb  # noqa: E402
from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio  # noqa: E402


def build_pipeline():
    snapshot_download("KwaiVGI/LivePortrait", local_dir=WEIGHTS,
                      allow_patterns=["insightface/*", "liveportrait/base_models/*",
                                      "liveportrait/retargeting_models/*", "liveportrait/landmark.onnx"])
    b = os.path.join(WEIGHTS, "liveportrait")
    inf = InferenceConfig(
        checkpoint_F=f"{b}/base_models/appearance_feature_extractor.pth",
        checkpoint_M=f"{b}/base_models/motion_extractor.pth",
        checkpoint_G=f"{b}/base_models/spade_generator.pth",
        checkpoint_W=f"{b}/base_models/warping_module.pth",
        checkpoint_S=f"{b}/retargeting_models/stitching_retargeting_module.pth",
    )
    crop = CropConfig(insightface_root=os.path.join(WEIGHTS, "insightface"),
                      landmark_ckpt_path=f"{b}/landmark.onnx")
    return LivePortraitPipeline(inference_cfg=inf, crop_cfg=crop)


def blink_curve(t, centers, dur=0.28):
    """0 = 眼睛張開，1 = 完全閉上（每次眨眼約 0.28 秒，閉得快、張得慢一點）"""
    v = 0.0
    for c in centers:
        x = (t - c) / dur
        if -0.4 <= x < 0:
            v = max(v, 1 + x / 0.4)
        elif 0 <= x <= 0.6:
            v = max(v, 1 - x / 0.6)
    return v


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=8.0, help="一輪循環幾秒")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--yaw", type=float, default=3.0, help="左右轉頭最大角度（度）")
    ap.add_argument("--pitch", type=float, default=1.5, help="上下點頭最大角度")
    ap.add_argument("--roll", type=float, default=1.5, help="歪頭最大角度")
    ap.add_argument("--blinks", default="1.6,4.9,5.4", help="哪幾秒眨眼（逗號分隔，可以連眨兩下）")
    ap.add_argument("--gaze", type=float, default=4.0, help="眼神左右飄的幅度（0 = 不飄）")
    ap.add_argument("--close-lips", type=int, default=1, help="1 = 嘴巴調成閉合（照片露齒、微張時讓對嘴更穩）")
    ap.add_argument("--lip-open-min", type=float, default=0.03, help="嘴巴張開程度超過這個才調（避免沒必要的變形）")
    a = ap.parse_args()

    lp = build_pipeline()
    w = lp.live_portrait_wrapper
    img = load_image_rgb(a.src)
    info = lp.cropper.crop_source_image(img, lp.cropper.crop_cfg)
    if info is None:
        raise SystemExit("角色圖裡找不到臉")
    I_s = w.prepare_source(info["img_crop_256x256"])
    x_info = w.get_kp_info(I_s)
    f_s = w.extract_feature_3d(I_s)
    x_s = w.transform_keypoint(x_info)
    lmk = info["lmk_crop"]
    mask = prepare_paste_back(w.inference_cfg.mask_crop, info["M_c2o"], dsize=(img.shape[1], img.shape[0]))
    R_s = get_rotation_matrix(x_info["pitch"], x_info["yaw"], x_info["roll"])
    eye_open = float(calc_eye_close_ratio(lmk[None]).mean())
    lip_open = float(calc_lip_close_ratio(lmk[None]).mean())
    print(f"找到臉；原本眼睛張開程度 {eye_open:.2f}，嘴巴張開程度 {lip_open:.3f}，姿勢 pitch {float(x_info['pitch']):.1f} "
          f"yaw {float(x_info['yaw']):.1f} roll {float(x_info['roll']):.1f}")
    lip_delta = None
    if a.close_lips and lip_open > a.lip_open_min:
        try:
            ratio = w.calc_combined_lip_ratio([0.0], lmk)
        except Exception:
            ratio = w.calc_combined_lip_ratio([[0.0]], lmk)
        lip_delta = w.retarget_lip(x_s, ratio)
        print("嘴巴是張開的：待機影片會調成閉合")

    n = int(round(a.seconds * a.fps))
    blinks = [float(s) for s in a.blinks.split(",") if s.strip()]
    H, W = img.shape[:2]
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "pipe:0",
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    tau = 2 * np.pi
    for i in range(n):
        t = i / a.fps
        p = t / a.seconds                                    # 0 → 1，頭尾都是原本的姿勢
        yaw = a.yaw * np.sin(tau * p)
        pitch = a.pitch * np.sin(tau * 2 * p + np.pi) * 0.5 + a.pitch * 0.5 * np.sin(tau * p)
        roll = a.roll * np.sin(tau * p + 0.0) * np.sin(tau * p * 0.5)   # 慢慢歪一下再回來
        R_d = get_rotation_matrix(x_info["pitch"] + pitch, x_info["yaw"] + yaw, x_info["roll"] + roll)
        R_new = (R_d @ R_s.permute(0, 2, 1)) @ R_s
        delta = x_info["exp"].clone()
        g = a.gaze * np.sin(tau * p) * 0.001                # 眼神跟著頭稍微往同方向看
        delta[0, 11, 0] += g
        delta[0, 15, 0] += g
        x_d = x_info["scale"] * (x_info["kp"] @ R_new + delta) + x_info["t"]
        b = blink_curve(t, blinks)
        if b > 0:
            ratio = eye_open * (1 - b) + 0.0 * b
            x_d = x_d + w.retarget_eye(x_s, w.calc_combined_eye_ratio([[ratio]], lmk))
        if lip_delta is not None:
            x_d = x_d + lip_delta
        x_d = w.stitching(x_s, x_d)
        out = w.parse_output(w.warp_decode(f_s, x_s, x_d)["out"])[0]
        frame = paste_back(out, info["M_c2o"], img, mask)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg 失敗")
    print(f"完成 → {a.out}（{n} 格，{a.seconds:.0f} 秒一輪；眨眼在 {blinks} 秒）")


if __name__ == "__main__":
    main()
