"""Wrapper neural TTS chạy OFFLINE cho Lab 4 (mục 4.6 / 5.6 của đề).

Dùng Piper (VITS, ONNX) – pip install piper-tts. Model giọng Việt phải được tải/đóng gói TRƯỚC giờ Lab
(đề cấm tải model ngẫu nhiên từ Internet trong giờ Lab). Đặt 2 file cạnh nhau:

    lab4_data/neural/vi_VN-vais1000-medium.onnx
    lab4_data/neural/vi_VN-vais1000-medium.onnx.json

(Voice "vi_VN-vais1000-medium" trong kho rhasspy/piper-voices trên Hugging Face.)
Có thể đổi đường dẫn bằng biến môi trường NEURAL_TTS_MODEL hoặc tham số model_path.

    from neural_tts_backend import synthesize
    y, sr, extra = synthesize("Mời khách hàng số hai ba đến quầy năm")
    # extra: latency_s, audio_s, rtf, load_time_s, model, mel (tính từ waveform, để vẽ)
"""
import os
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path("lab4_data/neural/vi_VN-vais1000-medium.onnx")
_VOICE = {}


def _get_voice(model_path):
    key = str(model_path)
    if key in _VOICE:
        return _VOICE[key], 0.0
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Chưa có model neural TTS tại '{model_path}'.\n"
            "Tải trước (ở nhà/trước giờ Lab) file .onnx và .onnx.json của voice vi_VN-vais1000-medium "
            "(Piper) rồi đặt vào lab4_data/neural/, hoặc đặt NEURAL_TTS_MODEL=<đường dẫn .onnx>.")
    from piper import PiperVoice  # pip install piper-tts
    t0 = time.perf_counter()
    voice = PiperVoice.load(str(model_path))
    load_time = time.perf_counter() - t0
    _VOICE[key] = voice
    return voice, load_time


def synthesize(text, model_path=None, target_sr=None):
    """Trả về (waveform float32 mono, sample_rate, extra). Latency đo từ lúc nhận text đến khi xong audio."""
    model_path = Path(model_path or os.environ.get("NEURAL_TTS_MODEL", DEFAULT_MODEL))
    voice, load_time = _get_voice(model_path)
    t0 = time.perf_counter()
    chunks = [c.audio_float_array for c in voice.synthesize(text)]
    y = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    latency = time.perf_counter() - t0
    sr = voice.config.sample_rate
    audio_s = len(y) / sr
    if target_sr and target_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    import librosa
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80), ref=np.max)
    extra = dict(latency_s=latency, audio_s=audio_s, rtf=latency / max(audio_s, 1e-9),
                 load_time_s=load_time, model=model_path.name, mel=mel,
                 mel_note="mel tính lại từ waveform (VITS không trả mel trung gian)",
                 model_size_mb=model_path.stat().st_size / 1e6)
    return y, sr, extra
