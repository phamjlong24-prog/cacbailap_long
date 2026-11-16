"""Script thu dữ liệu cho Lab 5 (CSE457): speech enhancement, speaker recognition, speaker separation.

Cấu trúc đầu ra (đúng mục 2.2 của đề), mono, 16 kHz, 16-bit PCM:
    lab5_data/clean/spk01_enroll_01.wav ... spk01_test_04.wav
    lab5_data/noise/fan.wav, cafeteria.wav
    lab5_data/separation/source_A.wav, source_B.wav

Cách dùng (mỗi người nói chạy lệnh `speaker` một lần trên máy):
    pip install sounddevice soundfile numpy

    python record_dataset.py speaker --name spk01            # 6 enroll + 4 test (mỗi file ~3 s)
    python record_dataset.py speaker --name spk02
    python record_dataset.py speaker --name spk03
    python record_dataset.py noise   --name fan --seconds 12 # thu tiếng quạt/máy lạnh... khi KHÔNG ai nói
    python record_dataset.py noise   --name cafeteria --seconds 12
    python record_dataset.py source  --name A --seconds 5    # người A nói liên tục ~5 s
    python record_dataset.py source  --name B --seconds 5    # NGƯỜI KHÁC nói (khác người nói A)

Gợi ý theo đề: ít nhất 3 người, mỗi người >= 4 enroll và >= 2 test (mặc định 6 + 4 để thống kê ổn định hơn);
file 2-4 s. Thu test vào một lúc khác với enroll. Nhóm tự thu phải có sự đồng ý của người tham gia và
không đẩy audio chưa được phép chia sẻ lên GitHub (xem .gitignore).
Nếu chưa có source_A/B, notebook tạm dùng 2 người nói khác nhau trong clean/ để tạo hỗn hợp.
"""
import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

FS = 16000
PROMPTS = [
    "Hôm nay trời nắng đẹp nên chúng tôi đi dạo quanh hồ.",
    "Xử lý âm thanh và tiếng nói là một môn học thú vị.",
    "Mời bạn ngồi xuống uống một chén trà nóng.",
    "Tôi đang làm bài thực hành số năm trên máy tính.",
    "Chiều nay lớp mình họp ở phòng hai trăm lẻ ba.",
    "Con mèo nhảy lên bàn và làm đổ cốc nước.",
    "Anh ấy mua ba quả cam và hai quả táo ở chợ.",
    "Xin hãy nói to và rõ ràng vào micro.",
    "Mùa thu Hà Nội có gió lạnh và hoa sữa thơm.",
    "Em sẽ nộp báo cáo trước hạn vào sáng thứ hai.",
]


def rec(seconds, prompt):
    input(prompt + "\n  -> bấm Enter để ghi...")
    y = sd.rec(int(seconds * FS), samplerate=FS, channels=1, dtype="float32")
    sd.wait()
    return y[:, 0]


def check_and_confirm(y):
    peak = float(np.max(np.abs(y)))
    warn = "  (CẢNH BÁO: clipping, nói nhỏ lại/để xa micro hơn)" if peak >= 0.99 else ""
    warn += "  (CẢNH BÁO: quá nhỏ, nói to hơn)" if peak < 0.05 else ""
    print(f"  đỉnh = {peak:.2f}{warn}")
    sd.play(y, FS)
    sd.wait()
    return input("  Giữ file này? [Enter = giữ, r = ghi lại] ").strip().lower() != "r"


def record_until_ok(seconds, prompt):
    while True:
        y = rec(seconds, prompt)
        if check_and_confirm(y):
            return y


def cmd_speaker(a):
    out = Path(a.out) / "clean"
    out.mkdir(parents=True, exist_ok=True)
    plan = [("enroll", k) for k in range(1, a.enroll + 1)] + [("test", k) for k in range(1, a.test + 1)]
    for i, (kind, k) in enumerate(plan):
        text = PROMPTS[i % len(PROMPTS)]
        y = record_until_ok(a.seconds, f"[{a.name} - {kind} {k}] đọc (~{a.seconds:.0f} s, giữ giọng tự nhiên): \"{text}\"")
        path = out / f"{a.name}_{kind}_{k:02d}.wav"
        sf.write(path, y, FS, subtype="PCM_16")
        print("  ->", path)


def cmd_noise(a):
    out = Path(a.out) / "noise"
    out.mkdir(parents=True, exist_ok=True)
    y = record_until_ok(a.seconds, f"[noise {a.name}] giữ im lặng, chỉ để tiếng nền ({a.seconds:.0f} s)")
    path = out / f"{a.name}.wav"
    sf.write(path, y, FS, subtype="PCM_16")
    print("  ->", path)


def cmd_source(a):
    out = Path(a.out) / "separation"
    out.mkdir(parents=True, exist_ok=True)
    y = record_until_ok(a.seconds, f"[source {a.name}] nói liên tục, ít ngắt ({a.seconds:.0f} s) – dùng NGƯỜI NÓI KHÁC nhau cho A và B")
    path = out / f"source_{a.name}.wav"
    sf.write(path, y, FS, subtype="PCM_16")
    print("  ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lab5_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("speaker"); p.add_argument("--name", required=True)
    p.add_argument("--enroll", type=int, default=6); p.add_argument("--test", type=int, default=4)
    p.add_argument("--seconds", type=float, default=3.5); p.set_defaults(fn=cmd_speaker)
    p = sub.add_parser("noise"); p.add_argument("--name", required=True)
    p.add_argument("--seconds", type=float, default=12.0); p.set_defaults(fn=cmd_noise)
    p = sub.add_parser("source"); p.add_argument("--name", required=True, choices=["A", "B"])
    p.add_argument("--seconds", type=float, default=5.0); p.set_defaults(fn=cmd_source)
    a = ap.parse_args()
    a.fn(a)
    print("Xong! Dữ liệu ở", Path(a.out).resolve())


if __name__ == "__main__":
    main()
