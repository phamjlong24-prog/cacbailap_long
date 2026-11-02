"""Script thu âm bộ dữ liệu cho Lab 3 (CSE457) – 5 lệnh: mở, đóng, lên, xuống, dừng.

Cách dùng:
    pip install sounddevice soundfile numpy
    python record_dataset.py                    # thu 8 lần/lệnh -> dataset/train (01-06) + dataset/test (07-08)
    python record_dataset.py --mismatch         # (tuỳ chọn, thí nghiệm E5) thu 2 lần/lệnh vào dataset/mismatch
                                                # trong điều kiện KHÁC: người khác nói, đứng xa micro, hoặc có nhạc/quạt nền
    python record_dataset.py --only mo dung     # chỉ thu lại một vài lệnh

Mỗi lần ghi 2 giây, 16 kHz, mono, 16-bit PCM.
Bấm Enter, đợi khoảng 0.3 s rồi nói MỘT lệnh, giữ im lặng trước và sau (đề yêu cầu 0.2-0.5 s silence).
Gợi ý: thu file 07-08 (test) vào một lúc khác với file 01-06 để test khách quan hơn.
"""
import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

FS = 16000
DURATION = 2.0
LABELS = {"mo": "mở", "dong": "đóng", "len": "lên", "xuong": "xuống", "dung": "dừng"}
N_TRAIN, N_TEST = 6, 2


def record_one(prompt):
    while True:
        input(prompt)
        y = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype="float32")
        sd.wait()
        y = y[:, 0]
        peak = float(np.max(np.abs(y)))
        warn = "  (CẢNH BÁO: clipping, nói nhỏ lại/để xa micro hơn)" if peak >= 0.99 else ""
        warn += "  (CẢNH BÁO: quá nhỏ, nói to hơn)" if peak < 0.05 else ""
        print(f"  đỉnh = {peak:.2f}{warn}")
        sd.play(y, FS); sd.wait()
        if input("  Giữ file này? [Enter = giữ, r = ghi lại] ").strip().lower() != "r":
            return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--mismatch", action="store_true", help="thu tập mismatch cho thí nghiệm E5")
    ap.add_argument("--only", nargs="*", default=None, help="chỉ thu các nhãn này")
    args = ap.parse_args()

    labels = {k: v for k, v in LABELS.items() if not args.only or k in args.only}
    root = Path(args.out)

    for lab, vn in labels.items():
        if args.mismatch:
            plan = [("mismatch", k) for k in range(1, 3)]
        else:
            plan = [("train", k) for k in range(1, N_TRAIN + 1)] + \
                   [("test", k) for k in range(N_TRAIN + 1, N_TRAIN + N_TEST + 1)]
        for split, k in plan:
            d = root / split / lab
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{lab}_{k:02d}.wav"
            y = record_one(f"[{split}] nói '{vn}' lần {k} - bấm Enter để ghi...")
            sf.write(path, y, FS, subtype="PCM_16")
            print("  ->", path)
    print("Xong! Dữ liệu ở", root.resolve())


if __name__ == "__main__":
    main()
