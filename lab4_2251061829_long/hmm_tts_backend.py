"""Backend HMM-based parametric synthesis (mini) cho Lab 4 – CSE457.

Ý tưởng (mục 3.7 của đề): HMM không dùng để nhận dạng mà chạy ở chế độ *generation*.
  - Mỗi từ có một HMM trái-sang-phải 5 state, mỗi state mang phân bố Gaussian (chéo) của
    o_t = [c_t, Δc_t, Δ²c_t]  (c_t = mel-cepstrum 24 chiều) và của log-F0, xác suất hữu thanh, duration.
  - Câu = nối các HMM từ; mỗi state được giữ số frame = duration trung bình × duration_scale.
  - Quỹ đạo tĩnh c* = argmax P(O|λ) với ràng buộc O = W c  (MLPG): nhờ đặc trưng động Δ, Δ² mà
    c* trơn giữa các state thay vì "bậc thang".
  - Vocoder source–filter: excitation (chuỗi xung theo F0 / nhiễu) được tạo hình phổ bằng đường bao
    phổ suy ra từ mel-cepstrum, tái tạo bằng STFT/OLA.

Huấn luyện từ chính inventory của bạn (units/*.wav) bằng segmental k-means (Viterbi alignment).

    from hmm_tts_backend import train_hmm_models, synthesize_hmm
    models = train_hmm_models("lab4_data/units", "lab4_data/hmm_demo/hmm_models.pkl")
    y, sr, tracks = synthesize_hmm("Mời khách hàng số 23 đến quầy 5", models=models)
"""
import pickle
import re
import unicodedata
from pathlib import Path

import librosa
import numpy as np
from scipy.fft import dct, idct

FS = 16000
WIN, HOP, NFFT = 400, 160, 512          # 25 ms / 10 ms
N_MELS, N_CEP, N_STATE = 40, 24, 5
F0_MIN, F0_MAX = 70.0, 400.0
FRAME_RATE = FS / HOP                    # 100 frame/s

DIGIT = {"0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
         "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"}


# ----------------------------------------------------------------------------- text
def normalize_text(text):
    """Chuẩn hoá miền hẹp: chữ thường, bỏ dấu câu, đọc từng chữ số. Trả về list token."""
    text = text.lower().replace(".", " ").replace(",", " ")
    out = []
    for tok in text.split():
        if tok.isdigit():
            out.extend(DIGIT[c] for c in tok)
        else:
            out.append(tok)
    return out


def ascii_label(word):
    """'mời' -> 'moi', 'đến' -> 'den' (khớp tên file trong units/)."""
    w = word.replace("đ", "d").replace("Đ", "D")
    w = unicodedata.normalize("NFD", w)
    w = "".join(ch for ch in w if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]", "", w.lower())


# ----------------------------------------------------------------------------- analysis
def _mel_centers():
    return librosa.mel_frequencies(n_mels=N_MELS, fmin=20, fmax=FS / 2)


def analyze(y):
    """y -> (mcep (T,N_CEP), lf0 (T,), voiced (T,) bool). Cùng số frame T."""
    S = librosa.feature.melspectrogram(y=y, sr=FS, n_fft=NFFT, win_length=WIN, hop_length=HOP,
                                       n_mels=N_MELS, power=2.0, fmin=20, fmax=FS / 2)
    logS = librosa.power_to_db(S, ref=1.0, top_db=None)               # (N_MELS, T) dB
    mcep = dct(logS, type=2, axis=0, norm="ortho")[:N_CEP].T          # (T, N_CEP)
    f0, vflag, _ = librosa.pyin(y, fmin=F0_MIN, fmax=F0_MAX, sr=FS, frame_length=1024,
                                hop_length=HOP)
    T = min(len(mcep), len(f0))
    mcep, f0, vflag = mcep[:T], f0[:T], vflag[:T].astype(bool)
    f0 = np.where(vflag & np.isfinite(f0), f0, np.nan)
    lf0 = np.log(f0)
    vflag = vflag & np.isfinite(lf0)
    return mcep, lf0, vflag


def deltas(c):
    """Δ, Δ² dạng sai phân (biên: lặp frame) – đúng ma trận W dùng trong MLPG."""
    cp = np.vstack([c[:1], c, c[-1:]])
    d1 = 0.5 * (cp[2:] - cp[:-2])
    d2 = cp[2:] - 2 * cp[1:-1] + cp[:-2]
    return d1, d2


def _obs(c):
    d1, d2 = deltas(c)
    return np.hstack([c, d1, d2])                                     # (T, 3*N_CEP)


# ----------------------------------------------------------------------------- training
def _viterbi_lr(logb):
    """Căn chỉnh trái-sang-phải: logb (T, N). Mỗi state >= 1 frame. Trả về state path."""
    T, N = logb.shape
    if T < N:                                      # quá ngắn: chia đều
        return np.minimum(np.arange(T) * N // max(T, 1), N - 1)
    NEG = -1e30
    delta = np.full((T, N), NEG)
    psi = np.zeros((T, N), dtype=int)
    delta[0, 0] = logb[0, 0]
    for t in range(1, T):
        for j in range(N):
            stay = delta[t - 1, j]
            move = delta[t - 1, j - 1] if j > 0 else NEG
            if move > stay:
                delta[t, j], psi[t, j] = move + logb[t, j], j - 1
            else:
                delta[t, j], psi[t, j] = stay + logb[t, j], j
    path = np.zeros(T, dtype=int)
    path[-1] = N - 1
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def train_word_hmm(takes, n_iter=6, n_state=N_STATE):
    """takes: list of (mcep, lf0, voiced). Segmental k-means. Trả về dict tham số."""
    obs = [_obs(m) for m, _, _ in takes]
    paths = [np.minimum(np.arange(len(o)) * n_state // len(o), n_state - 1) for o in obs]
    D = obs[0].shape[1]
    mean = np.zeros((n_state, D))
    var = np.ones((n_state, D))
    for it in range(n_iter):
        allo, allp = np.vstack(obs), np.concatenate(paths)
        gvar = allo.var(axis=0) + 1e-3
        for s in range(n_state):
            x = allo[allp == s]
            if len(x) == 0:
                x = allo
            mean[s] = x.mean(axis=0)
            var[s] = np.maximum(x.var(axis=0), 0.05 * gvar)
        paths = []
        for o in obs:
            logb = np.stack([-0.5 * np.sum(np.log(2 * np.pi * var[s]) + (o - mean[s]) ** 2 / var[s], axis=1)
                             for s in range(n_state)], axis=1)
            paths.append(_viterbi_lr(logb))
    # log-F0, voicing, duration theo state
    lf0_mean, lf0_var, vprob, dur = (np.zeros(n_state) for _ in range(4))
    all_lf0 = np.concatenate([l[v] for _, l, v in takes if v.any()] or [np.log([120.0])])
    for s in range(n_state):
        vs, vc = [], 0
        tot = 0
        for (m, l, v), p in zip(takes, paths):
            sel = p == s
            tot += sel.sum()
            vc += (v & sel).sum()
            vs.append(l[v & sel])
        vs = np.concatenate(vs) if vs else np.array([])
        vprob[s] = vc / max(tot, 1)
        lf0_mean[s] = vs.mean() if len(vs) >= 2 else all_lf0.mean()
        lf0_var[s] = max(vs.var() if len(vs) >= 2 else all_lf0.var(), 0.003)
        dur[s] = np.mean([(p == s).sum() for p in paths])
    return dict(mean=mean, var=var, lf0_mean=lf0_mean, lf0_var=lf0_var, vprob=vprob,
                dur=np.maximum(dur, 1.0), n_takes=len(takes))


def _load_trim(path):
    y, _ = librosa.load(str(path), sr=FS, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    return y


def train_hmm_models(units_dir, out_path=None, verbose=True):
    """Huấn luyện 1 HMM/từ từ units_dir/<nhãn>_NN.wav. Trả về dict {nhãn_ascii: params}."""
    units_dir = Path(units_dir)
    by_label = {}
    for f in sorted(units_dir.glob("*.wav")):
        lab = f.stem.rsplit("_", 1)[0]
        by_label.setdefault(lab, []).append(f)
    models = {}
    for lab, files in by_label.items():
        takes = [analyze(_load_trim(f)) for f in files]
        models[lab] = train_word_hmm(takes)
        if verbose:
            print(f"  HMM '{lab}': {len(files)} takes, dur/state(frame) = {np.round(models[lab]['dur'], 1)}")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fh:
            pickle.dump(models, fh)
    return models


def load_hmm_models(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ----------------------------------------------------------------------------- generation
def _window_matrix(T):
    """W (3T x T): [static; delta; accel] theo đúng định nghĩa ở deltas()."""
    I = np.eye(T)
    idx = np.arange(T)
    prev, nxt = np.clip(idx - 1, 0, T - 1), np.clip(idx + 1, 0, T - 1)
    Wd, Wa = np.zeros((T, T)), np.zeros((T, T))
    for t in range(T):
        Wd[t, nxt[t]] += 0.5
        Wd[t, prev[t]] -= 0.5
        Wa[t, nxt[t]] += 1.0
        Wa[t, t] -= 2.0
        Wa[t, prev[t]] += 1.0
    return np.vstack([I, Wd, Wa])


def mlpg(mu3, var3, W=None):
    """mu3, var3: (T, 3) -> quỹ đạo tĩnh (T,) = argmax P(W c | μ, Σ)."""
    T = mu3.shape[0]
    W = _window_matrix(T) if W is None else W
    mu = np.concatenate([mu3[:, 0], mu3[:, 1], mu3[:, 2]])
    prec = 1.0 / np.concatenate([var3[:, 0], var3[:, 1], var3[:, 2]])
    A = W.T @ (W * prec[:, None])
    b = W.T @ (prec * mu)
    return np.linalg.solve(A + 1e-9 * np.eye(T), b)


def _expand(tokens, models, duration_scale, pause_frames):
    """Mở rộng chuỗi từ -> tham số theo frame."""
    mean, var, lm, lv, vp = [], [], [], [], []
    sid, wid, sil = [], [], []
    word_start = []
    for w_i, tok in enumerate(tokens):
        lab = ascii_label(tok)
        if lab not in models:
            raise KeyError(f"Không có HMM cho từ '{tok}' (nhãn '{lab}'). Có: {sorted(models)}")
        m = models[lab]
        word_start.append(len(sid))
        for s in range(N_STATE):
            n = max(1, int(round(m["dur"][s] * duration_scale)))
            for _ in range(n):
                mean.append(m["mean"][s]); var.append(m["var"][s])
                lm.append(m["lf0_mean"][s]); lv.append(m["lf0_var"][s]); vp.append(m["vprob"][s])
                sid.append(w_i * N_STATE + s); wid.append(w_i); sil.append(False)
        if w_i < len(tokens) - 1 and pause_frames > 0:       # khoảng lặng ngắn giữa các từ
            for _ in range(pause_frames):
                mean.append(mean[-1]); var.append(var[-1]); lm.append(lm[-1]); lv.append(lv[-1])
                vp.append(0.0); sid.append(sid[-1]); wid.append(w_i); sil.append(True)
    return (np.array(mean), np.array(var), np.array(lm), np.array(lv), np.array(vp),
            np.array(sid), np.array(wid), np.array(sil), word_start)


def generate_params(tokens, models, duration_scale=1.0, f0_scale=1.0, pause_frames=4,
                    use_dynamic=True):
    """Sinh quỹ đạo tham số cho câu. use_dynamic=False -> chỉ dùng mean tĩnh (đường bậc thang)."""
    mean, var, lm, lv, vp, sid, wid, sil, word_start = _expand(tokens, models, duration_scale, pause_frames)
    T = len(mean)
    W = _window_matrix(T)
    static_mean = mean[:, :N_CEP]
    if use_dynamic:
        c = np.zeros((T, N_CEP))
        for d in range(N_CEP):
            mu3 = np.stack([mean[:, d], mean[:, N_CEP + d], mean[:, 2 * N_CEP + d]], axis=1)
            v3 = np.stack([var[:, d], var[:, N_CEP + d], var[:, 2 * N_CEP + d]], axis=1)
            c[:, d] = mlpg(mu3, v3, W)
    else:
        c = static_mean.copy()
    voiced = (vp > 0.5) & (~sil)
    # log-F0: frame vô thanh có phương sai rất lớn -> được nội suy từ lân cận (MLPG)
    mu3 = np.stack([lm, np.zeros(T), np.zeros(T)], axis=1)
    v3 = np.stack([np.where(voiced, lv, 1e4), np.full(T, 0.004), np.full(T, 0.01)], axis=1)
    lf0 = mlpg(mu3, v3, W) if use_dynamic else lm.copy()
    f0 = np.exp(lf0) * f0_scale
    f0 = np.clip(f0, F0_MIN * 0.8, F0_MAX * 1.5)
    return dict(c=c, c_static_mean=static_mean, f0=f0, voiced=voiced, silence=sil,
                states=sid, word_idx=wid, word_start=np.array(word_start), tokens=list(tokens))


def vocode(c, f0, voiced, silence, seed=0):
    """Vocoder source–filter đơn giản: excitation (xung/nhiễu) × đường bao phổ từ mel-cepstrum."""
    rng = np.random.default_rng(seed)
    T = len(c)
    n = T * HOP
    frames_t = np.arange(T) * HOP
    f0_s = np.interp(np.arange(n), frames_t, f0)
    v_s = np.interp(np.arange(n), frames_t, voiced.astype(float))
    phase = np.cumsum(f0_s / FS)
    pulses = np.diff(np.floor(phase), prepend=0.0) > 0
    exc = v_s * pulses * np.sqrt(FS / np.maximum(f0_s, 1.0)) + (1 - v_s) * rng.standard_normal(n) * 0.3
    E = librosa.stft(exc, n_fft=NFFT, hop_length=HOP, win_length=WIN, window="hann", center=True)
    E = E / (np.sqrt(np.mean(np.abs(E) ** 2, axis=0, keepdims=True)) + 1e-9)    # phổ phẳng, giữ pha
    # đường bao: mcep -> log-mel (dB) -> nội suy lên thang tần số tuyến tính
    pad = np.zeros((T, N_MELS))
    pad[:, :N_CEP] = c
    logmel_db = idct(pad, type=2, axis=1, norm="ortho")                           # (T, N_MELS)
    fr = librosa.fft_frequencies(sr=FS, n_fft=NFFT)
    mc = _mel_centers()
    env_db = np.stack([np.interp(fr, mc, logmel_db[t]) for t in range(T)], axis=1)  # (bins, T)
    env = 10 ** (env_db / 20.0)
    gain = np.where(silence, 0.02, 1.0)
    K = min(E.shape[1], T)
    Y = E[:, :K] * env[:, :K] * gain[None, :K]
    y = librosa.istft(Y, hop_length=HOP, win_length=WIN, window="hann", length=n)
    act = np.repeat(~silence, HOP)[: len(y)]
    rms = np.sqrt(np.mean(y[act] ** 2) + 1e-12) if act.any() else 1.0
    y = y * (0.08 / rms)
    return np.clip(y, -0.99, 0.99).astype(np.float32)


def synthesize_hmm(text, models=None, models_path=None, duration_scale=1.0, f0_scale=1.0,
                   pause_frames=4, use_dynamic=True):
    """API theo đề: y, sr, tracks = synthesize_hmm("Mời khách hàng số 23 đến quầy 5")."""
    if models is None:
        models = load_hmm_models(models_path or "lab4_data/hmm_demo/hmm_models.pkl")
    tokens = normalize_text(text) if isinstance(text, str) else list(text)
    p = generate_params(tokens, models, duration_scale, f0_scale, pause_frames, use_dynamic)
    y = vocode(p["c"], p["f0"], p["voiced"], p["silence"])
    tracks = dict(p)
    tracks["f0_hz"] = np.where(p["voiced"], p["f0"], 0.0)
    tracks["frame_rate"] = FRAME_RATE
    return y, FS, tracks
