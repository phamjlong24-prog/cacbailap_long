"""Script thu inventory cho Lab 4 (CSE457) – miền hẹp "thông báo xếp hàng".

16 đơn vị: mời, khách, hàng, số, không, một, hai, ba, bốn, năm, sáu, bảy, tám, chín, đến, quầy.
Mỗi đơn vị thu 3 bản (take) -> lab4_data/units/<nhãn>_01..03.wav (mono, 16 kHz, 16-bit PCM).

Cách dùng:
    pip install sounddevice soundfile numpy
    python record_units.py                  # thu tất cả
    python record_units.py --only moi hai   # thu lại một vài đơn vị
    python record_units.py --takes 2        # số take/đơn vị (mặc định 3)

Yêu cầu của đề: cùng microphone, cùng khoảng cách; mỗi từ chừa khoảng 100-200 ms im lặng ở đầu/cuối
(notebook sẽ tự cắt/trim). Đừng nói quá to để tránh clipping. Nói TỰ NHIÊN, mỗi take hơi khác nhau
một chút (để unit selection có lựa chọn) – nhưng cùng một giọng, cùng tốc độ vừa phải.
"""
import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

FS = 16000
DURATION = 1.5          # giây cho mỗi lần ghi
UNITS = {
    "moi": "mời", "khach": "khách", "hang": "hàng", "so": "số",
    "khong": "không", "mot": "một", "hai": "hai", "ba": "ba", "bon": "bốn",
    "nam": "năm", "sau": "sáu", "bay": "bảy", "tam": "tám", "chin": "chín",
    "den": "đến", "quay": "quầy",
}


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
        sd.play(y, FS)
        sd.wait()
        if input("  Giữ file này? [Enter = giữ, r = ghi lại] ").strip().lower() != "r":
            return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lab4_data/units")
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="chỉ thu các nhãn này (vd: moi hai)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for lab, vn in UNITS.items():
        if args.only and lab not in args.only:
            continue
        for k in range(1, args.takes + 1):
            y = record_one(f"nói '{vn}' (take {k}/{args.takes}) - bấm Enter, đợi ~0.2 s rồi nói...")
            path = out / f"{lab}_{k:02d}.wav"
            sf.write(path, y, FS, subtype="PCM_16")
            print("  ->", path)
    print("Xong! Inventory ở", out.resolve())


if __name__ == "__main__":
    main()
