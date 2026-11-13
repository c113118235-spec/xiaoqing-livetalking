"""用新的設定重做一個自訂學伴（不用使用者重新上傳照片）：用留下來的 canvas.jpg 重跑 LivePortrait 和臉框。

在 xiaoqing 容器裡跑（主機上的小幫手 tools/lp_worker.sh 要開著）：
  docker exec xiaoqing python /app/tools/rebuild_custom.py custom_xxxxxxxxxxxxxxxx --lp "--preset natural --seconds 12"
  docker restart xiaoqing                      # 人物載入後會被快取，重開才會用新的
還原成重做前的版本：
  docker exec xiaoqing python /app/tools/rebuild_custom.py custom_xxxxxxxxxxxxxxxx --restore
"""
import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("avatar_id")
    ap.add_argument("--lp", default="--preset natural --seconds 12", help="給 idle_gen.py 的參數")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--face-max", type=int, default=150, help="臉最寬幾像素（太大嘴型會糊）；0 = 不縮")
    a = ap.parse_args()
    from server import gf_custom as g
    if not g.valid_id(a.avatar_id):
        sys.exit("id 格式不對（custom_ 加 16 個英數字）")
    cdir, adir = g._dir(a.avatar_id), os.path.join(g.AVATAR_DIR, a.avatar_id)
    bk = os.path.join(cdir, "avatar_backup")
    if a.restore:
        if not os.path.isdir(bk):
            sys.exit("沒有備份")
        shutil.rmtree(adir, ignore_errors=True)
        shutil.copytree(bk, adir)
        if os.path.isfile(os.path.join(cdir, "canvas_orig.jpg")):
            shutil.copy2(os.path.join(cdir, "canvas_orig.jpg"), os.path.join(cdir, "canvas.jpg"))
        print("已還原；docker restart xiaoqing 後生效")
        return
    canvas = os.path.join(cdir, "canvas.jpg")
    if not os.path.isfile(canvas):
        sys.exit(f"找不到 {canvas}（太舊的學伴沒有留畫布，只能請使用者重新上傳）")
    if not g.worker_alive():
        sys.exit("主機上的小幫手沒開：nohup ~/livetalking/tools/lp_worker.sh > ~/livetalking/data/custom/lp_worker.log 2>&1 &")
    if os.path.isdir(adir) and not os.path.isdir(bk):
        shutil.copytree(adir, bk)                    # 只備份第一次（原始版）
        print(f"原本的版本備份在 {bk}")
    if not os.path.isfile(os.path.join(cdir, "canvas_orig.jpg")):
        shutil.copy2(canvas, os.path.join(cdir, "canvas_orig.jpg"))
    else:
        shutil.copy2(os.path.join(cdir, "canvas_orig.jpg"), canvas)   # 每次都從原始畫布開始縮
    if a.face_max > 0:
        fw = g.shrink_canvas(canvas, a.face_max)
        print(f"臉寬 {fw} px" + (f" → 縮成約 {a.face_max} px" if fw and fw > a.face_max else "（不用縮）"))
    with open(os.path.join(cdir, "lp_args"), "w") as f:
        f.write(a.lp)
    os.chmod(os.path.join(cdir, "lp_args"), 0o666)
    t0 = time.time()
    print(f"1/2 LivePortrait 重做待機影片（{a.lp}）…")
    video = g._animate_by_worker(a.avatar_id)
    print(f"    完成（{time.time() - t0:.0f} 秒）")
    print("2/2 重新找臉框（固定大小＋平滑）…")
    tmp = a.avatar_id + "_rebuild"
    shutil.rmtree(os.path.join(g.AVATAR_DIR, tmp), ignore_errors=True)
    from avatars.wav2lip.genavatar import generate_avatar
    generate_avatar(video_path=video, avatar_id=tmp, save_path=g.AVATAR_DIR, img_size=g.IMG_SIZE, stable=True)
    shutil.rmtree(adir, ignore_errors=True)
    os.rename(os.path.join(g.AVATAR_DIR, tmp), adir)
    print(f"完成（共 {time.time() - t0:.0f} 秒）。docker restart xiaoqing 後，重新整理網頁選這個學伴。")


if __name__ == "__main__":
    main()
