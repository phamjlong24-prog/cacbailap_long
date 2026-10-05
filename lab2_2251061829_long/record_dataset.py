"""Script thu âm bộ dữ liệu cho Lab 2 (CSE457).

Cách dùng:
    pip install sounddevice soundfile numpy
    python record_dataset.py                 # ghi vào dataset/
    python record_dataset.py --out dataset_speaker2   # người nói thứ hai (thí nghiệm E4)

Với mỗi từ, script ghi 5 lần, mỗi lần 2 giây, 16 kHz, mono.
Bấm Enter, đợi ~0.3 s rồi nói MỘT từ, giữ im lặng trước và sau từ.
"""
import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

FS = 16000
DURATION = 2.0
LABELS = {"khong": "không", "mot": "một", "hai": "hai", "ba": "ba", "bon": "bốn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    for lab, vn in LABELS.items():
        d = Path(args.out) / lab
        d.mkdir(parents=True, exist_ok=True)
        for k in range(1, args.reps + 1):
            path = d / f"{lab}_{k:02d}.wav"
            while True:
                input(f"[{lab}] nói '{vn}' lần {k}/{args.reps} - bấm Enter để ghi...")
                y = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype="float32")
                sd.wait()
                y = y[:, 0]
                print(f"  đỉnh = {np.max(np.abs(y)):.2f}", "(CẢNH BÁO: clipping)" if np.max(np.abs(y)) >= 0.99 else "")
                sd.play(y, FS); sd.wait()
                if input("  Giữ file này? [Enter = giữ, r = ghi lại] ").strip().lower() != "r":
                    sf.write(path, y, FS)
                    break
    print("Xong! Dữ liệu ở", args.out)


if __name__ == "__main__":
    main()
