# CSE457 – Lab 3: Nhận dạng tiếng nói nâng cao với HMM

- **Họ tên:** Phạm Duy Long – **MSSV:** 2251061829 – **Lớp:** 64CNTT1
- Môn Xử lý âm thanh và tiếng nói, Trường Đại học Thủy lợi.

Nhận dạng 5 lệnh tiếng Việt tách rời **mở, đóng, lên, xuống, dừng** bằng whole-word Gaussian HMM trái-sang-phải:

`WAV → cắt silence → MFCC 13 + Δ + Δ² (39-D) + CMVN → 1 HMM 5 state/từ (Baum-Welch) → argmax log P(X|λ) → (mở rộng) bigram re-ranking`

## Cấu trúc

```
├── Lab03_2251061829.ipynb   # notebook chính, mục A–H + thí nghiệm E1–E6 + trả lời câu hỏi
├── record_dataset.py        # script thu âm 5 lệnh × 8 lần (+ tập mismatch tuỳ chọn)
├── lexicon.txt              # lexicon mini: WORD PHONE_1 ... PHONE_K
├── corpus.txt               # chuỗi lệnh để đếm unigram/bigram (phần G)
├── requirements.txt
├── dataset/                 # WAV mono 16 kHz, 16-bit
│   ├── train/<nhãn>/<nhãn>_01..06.wav
│   ├── test/<nhãn>/<nhãn>_07..08.wav
│   └── mismatch/<nhãn>/...  # tuỳ chọn (E5): người khác nói / đứng xa micro / có nhiễu nền
├── models/                  # 5 HMM đã huấn luyện (pickle) – sinh ra khi chạy
├── figures/                 # hình – sinh ra khi chạy
└── results/                 # bảng CSV – sinh ra khi chạy
```

Nhãn thư mục: `mo`, `dong`, `len`, `xuong`, `dung`.

## Cách chạy

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Thu âm: 8 lần/lệnh, tự chia 01–06 vào train, 07–08 vào test
python record_dataset.py
# (tuỳ chọn, cho E5) thu thêm 2 lần/lệnh trong điều kiện khác
python record_dataset.py --mismatch

# 2. Chạy notebook từ đầu đến cuối
jupyter notebook Lab03_2251061829.ipynb     # Kernel → Restart & Run All
```

Đã thử với numpy 2.x, librosa 1.0, hmmlearn 0.3.3, scikit-learn 1.9. Toàn bộ notebook chạy khoảng 30 giây trên CPU.

## Tham số baseline

| Tham số | Giá trị |
|---|---|
| Fs, frame / hop | 16 kHz, 25 ms / 10 ms, Hamming, NFFT 512 |
| Đặc trưng | 13 MFCC (26 mel) + Δ + Δ² = 39 chiều, CMVN theo utterance |
| Cắt silence | `librosa.effects.trim(top_db=30)` + chừa 30 ms mỗi phía |
| HMM | 5 state trái-sang-phải, π = [1,0,0,0,0], a_ii = 0.7, a_i,i+1 = 0.3, state cuối a = 1 |
| Phát xạ | 1 Gaussian diagonal/state, `min_covar = 1e-3` |
| Khởi tạo | means/vars theo phân đoạn đều (flat start); Baum-Welch cập nhật `tmc`, tối đa 80 vòng, tol 1e-3 |
| Nhận dạng | argmax log P(X\|λ_w) (thuật toán Forward) |
| Bigram | add-k, k = 0.5; Score = log P(X\|λ)/T + λ_LM · log P(w\|w_prev) |

## Nhận dạng một file mới

```python
pred, scores = recognize("duong_dan/file.wav", models)
print(pred, scores)                     # nhãn + log-likelihood với 5 model
F = extract_feat("duong_dan/file.wav")
print(recognize_with_reject(F, models)) # trả "unknown" nếu dưới ngưỡng θ
```

## Kết quả

Sau khi chạy với dữ liệu thu âm: xem `results/experiments_summary.csv` (E1–E6), `results/E_dtw_vs_hmm.csv`, `figures/` và ô "TÓM TẮT KẾT QUẢ" cuối notebook.
