# CSE457 – Lab 2: Nhận dạng từ đơn bằng MFCC + DTW

Bài thực hành môn Xử lý âm thanh và tiếng nói (Trường Đại học Thủy lợi).
Nhận dạng 5 từ tiếng Việt tách rời (*không, một, hai, ba, bốn*) bằng pipeline tự cài đặt:

`WAV → endpoint detection (log-energy + ZCR) → pre-emphasis → Hamming → MFCC (13, 24 mel) → DTW tự cài → nearest-template`

Không dùng dịch vụ ASR trực tuyến hay mô hình end-to-end.

## Cấu trúc

```
├── Lab2_MSSV.ipynb      # notebook chính (đổi MSSV thành mã số của bạn)
├── record_dataset.py    # script thu âm 5 từ x 5 lần
├── requirements.txt
├── dataset/             # file WAV mono 16 kHz: dataset/<nhan>/<nhan>_01.wav ... _05.wav
│   ├── khong/ mot/ hai/ ba/ bon/
├── figures/             # hình xuất ra từ notebook
├── results.csv          # kết quả nhận dạng trên tập test
└── experiments.csv      # bảng thí nghiệm E1–E3
```

## Cách chạy

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Thu dữ liệu (hoặc tự chép file WAV vào dataset/ đúng cấu trúc)
python record_dataset.py

# 2. Chạy notebook từ đầu đến cuối
jupyter notebook Lab2_MSSV.ipynb     # Kernel → Restart & Run All
```

Chia dữ liệu: 3 file đầu mỗi từ làm template, các file còn lại làm test (không trùng nhau).

## Nhận dạng một file mới

Trong notebook (mục F):

```python
pred, scores = recognize_file("duong_dan/file.wav", templates)
print(pred, scores)   # nhãn dự đoán + khoảng cách DTW tới từng lớp
```

## Tham số baseline

| Tham số | Giá trị |
|---|---|
| Fs | 16 kHz |
| Frame / Hop | 25 ms (400 mẫu) / 10 ms (160 mẫu) |
| Window | Hamming |
| Pre-emphasis | α = 0.97 |
| NFFT | 512 |
| Mel filters / MFCC | 24 / 13 (+ CMN) |
| Endpoint | ngưỡng = đỉnh − 30 dB (và ≥ nền + 10 dB), ZCR refine, margin 50 ms |
| DTW | Euclid, 3 bước cục bộ, chuẩn hoá theo độ dài path |

## Kết quả

Điền accuracy và nhận xét sau khi chạy với dữ liệu của bạn (xem `results.csv`, `experiments.csv`, `figures/`).
