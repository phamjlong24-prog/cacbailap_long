# Lab 5 – Speech Enhancement, Speaker Recognition, Speaker Separation (CSE457)

Phạm Duy Long – MSSV 2251061829 – 64CNTT1

## Nội dung
- `Lab05_2251061829.ipynb`: toàn bộ thí nghiệm (4.1–4.8, TN1–TN8, 12 câu hỏi). Tham số nằm ở cell `CFG`.
- `record_dataset.py`: thu dữ liệu đúng cấu trúc đề.
- `requirements.txt`

## Thu dữ liệu (không commit audio)
```
pip install -r requirements.txt
python record_dataset.py speaker --name spk01   # lặp lại cho spk02, spk03 (6 enroll + 4 test)
python record_dataset.py noise --name fan --seconds 12
python record_dataset.py noise --name cafeteria --seconds 12
python record_dataset.py source --name A --seconds 5
python record_dataset.py source --name B --seconds 5   # người nói KHÁC A
```
Kết quả nằm ở `lab5_data/{clean,noise,separation}/`.

## Chạy
`jupyter notebook Lab05_2251061829.ipynb` → Restart & Run All. Kết quả sinh ra ở `results/` (audio, figures, csv).

## Việc cần làm tay
Điền các chỗ "(điền sau khi nghe/xem bảng)" bằng nhận xét từ kết quả thật của bạn, rồi viết báo cáo 4–8 trang.
