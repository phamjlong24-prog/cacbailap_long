# CSE457 – Lab 4: Tổng hợp tiếng nói (TTS)

- **Họ tên:** Phạm Duy Long – **MSSV:** 2251061829 – **Lớp:** 64CNTT1
- Môn Xử lý âm thanh và tiếng nói, Trường Đại học Thủy lợi.

Mini TTS miền hẹp **"thông báo xếp hàng"** (câu mẫu *"Mời khách hàng số 23 đến quầy 5."*) và so sánh ba hướng tổng hợp:

1. **Ghép nối** word-unit: text normalization → inventory → unit selection (target cost + join cost, quy hoạch động) → cross-fade.
2. **Tham số HMM**: HMM từng từ (5 state, Gaussian trên mel-cepstrum + Δ + Δ² + log-F0 + duration) → sinh quỹ đạo bằng MLPG → vocoder source–filter.
3. **TTS học sâu** pretrained chạy offline (Piper / VITS, ONNX).

## Cấu trúc

```
├── Lab04_2251061829.ipynb   # notebook chính: mục A–G + thí nghiệm TN1–TN5 + trả lời 10 câu hỏi
├── record_units.py          # script thu inventory: 16 từ x 3 take
├── hmm_tts_backend.py       # backend HMM: train_hmm_models(), synthesize_hmm()  (tự huấn luyện từ inventory)
├── neural_tts_backend.py    # wrapper neural TTS offline (Piper): synthesize()
├── requirements.txt
├── lab4_data/               # (không đưa lên repo)
│   ├── units/               # moi_01.wav ... quay_03.wav   (mono, 16 kHz, 16-bit PCM)
│   ├── neural/              # vi_VN-vais1000-medium.onnx (+ .onnx.json)
│   ├── hmm_demo/            # hmm_models.pkl – sinh ra khi chạy
│   └── metadata.csv         # sinh ra khi chạy (label, speaker, duration, RMS, path)
├── audio_output/            # raw_concat.wav, best_concat.wav, hmm_tts.wav, neural_tts.wav, xfade/, prosody/, blind/
├── figures/                 # hình sinh ra khi chạy
├── results/                 # bảng CSV sinh ra khi chạy
└── ratings.csv              # (mẫu) kết quả nghe mù, điền tay
```

Nhãn unit: `moi khach hang so khong mot hai ba bon nam sau bay tam chin den quay` (16 từ, mỗi từ ≥ 2 take).

## Cách chạy

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Thu inventory (cùng micro, cùng khoảng cách, chừa ~0.2 s lặng đầu/cuối mỗi lần ghi)
python record_units.py

# 2. (Làm TRƯỚC giờ Lab, khi có Internet) tải model neural TTS Việt vào lab4_data/neural/
python -m piper.download_voices vi_VN-vais1000-medium --download-dir lab4_data/neural
#   -> kiểm tra có 2 file: vi_VN-vais1000-medium.onnx và vi_VN-vais1000-medium.onnx.json
#   (đổi voice khác: đặt biến môi trường NEURAL_TTS_MODEL=<đường dẫn .onnx>)

# 3. Chạy notebook từ đầu đến cuối
jupyter notebook Lab04_2251061829.ipynb             # Kernel → Restart & Run All
```

Nếu chưa có model neural, notebook vẫn chạy hết và ghi rõ phần "Neural TTS chưa chạy"; bảng so sánh khi đó chỉ có 2 phương pháp.

## Tham số baseline

| Tham số | Giá trị |
|---|---|
| Fs, định dạng | 16 kHz, mono, 16-bit PCM |
| Cắt unit | `librosa.effects.trim(top_db=35)` + lề 10 ms, trượt biên về zero-crossing (±5 ms), RMS chuẩn hoá 0.08 |
| Target cost | $\|\ln(d/d_t)\|/0.15 + 0.5\cdot\|12\log_2(F0/F0_t)\|/2$; từ cuối câu: duration ×1.15, F0 ×0.90 |
| Join cost | log-phổ 25 ms hai phía biên + 0.5·\|ΔRMS\| + 0.5·\|ΔF0\| (mỗi thành phần chuẩn hoá theo σ), λ = 1 |
| Cross-fade | 15 ms (thí nghiệm 0/5/15/30 ms) |
| Prosody | F0 bằng pyin (70–400 Hz); time-stretch 0.8/1.0/1.2×; pitch ±2 semitone bằng librosa và TD-PSOLA tự cài |
| HMM | 5 state trái-sang-phải/từ; mel-cepstrum 24 chiều (40 mel) + Δ + Δ²; Gaussian chéo; segmental k-means; MLPG |
| Neural | Piper `vi_VN-vais1000-medium` (VITS, ONNX), chạy CPU offline |

## Nghe mù và đánh giá chủ quan

Notebook tạo `audio_output/blind/S01.wav…` (tên ngẫu nhiên, cùng loudness; khoá ở `results/blind_key.csv`, **không đưa cho người nghe**). Mỗi người nghe điền vào `ratings.csv`: `listener, code, heard_text (nghe được gì), naturalness_1_5, continuity_1_5`. Chạy lại ô cuối mục G để cập nhật cột Intelligibility/MOS trong `results/G_comparison.csv`.

## Kết quả

Sau khi chạy với dữ liệu thu âm: xem `results/` (TN1–TN4, so sánh G), `figures/` và ô "TÓM TẮT KẾT QUẢ LAB 4" cuối notebook.
