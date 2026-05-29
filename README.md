# 🧠 Emotion Recognition — EfficientNetV2S
**Deteksi 4 Emosi Wajah: Angry | Happy | Neutral | Sad**
Target Akurasi: ≥ 80% | Real-time Webcam | TensorFlow/Keras

---

## 📁 Struktur File

```
emotion_project/
├── train_emotion.py       ← Script training utama
├── realtime_emotion.py    ← Deteksi real-time webcam
├── prepare_dataset.py     ← Helper siapkan dataset FER2013
├── dataset/               ← Folder dataset kamu
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   ├── neutral/
│   │   └── sad/
│   ├── val/  (struktur sama)
│   └── test/ (struktur sama)
└── models/                ← Model hasil training (auto-dibuat)
    ├── emotion_model_final.keras
    └── emotion_model.tflite
```

---

## 🚀 Langkah-Langkah

### 1. Install Dependencies
```bash
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pillow
```

### 2. Siapkan Dataset
**Opsi A — FER2013 (paling mudah):**
```bash
# Download dari: https://www.kaggle.com/datasets/msambare/fer2013
# Setelah download & extract:
python prepare_dataset.py --src ./fer2013 --dst ./dataset
```

**Opsi B — RAF-DB (akurasi lebih tinggi ~86%):**
```
Daftar di: http://www.whdeng.cn/RAF/model2.html
Susun manual ke folder dataset/train/val/test/[angry|happy|neutral|sad]/
```

**Target jumlah data minimum:**
| Kelas   | Train | Val  | Test |
|---------|-------|------|------|
| Angry   | ≥1500 | ≥200 | ≥200 |
| Happy   | ≥2000 | ≥300 | ≥300 |
| Neutral | ≥1500 | ≥200 | ≥200 |
| Sad     | ≥1500 | ≥200 | ≥200 |

### 3. Training
```bash
python train_emotion.py
```
Training berjalan dalam **2 fase**:
- **Fase 1 (Warmup, ~10 epoch):** Base EfficientNetV2S dibekukan, hanya head yang dilatih
- **Fase 2 (Fine-tune, ~40 epoch):** 60 layer terakhir EfficientNetV2S di-unfreeze

### 4. Jalankan Real-time Detection
```bash
# Webcam default (index 0)
python realtime_emotion.py

# Webcam index lain
python realtime_emotion.py --source 1

# Video file
python realtime_emotion.py --source ./video.mp4

# Simpan output ke file
python realtime_emotion.py --save
```

**Kontrol keyboard:**
| Tombol | Fungsi |
|--------|--------|
| Q / ESC | Keluar |
| S | Screenshot |
| P | Pause/Resume |
| R | Reset smoothing buffer |

---

## ⚙️ Konfigurasi Penting (di train_emotion.py)

```python
CONFIG = {
    "img_size"       : (224, 224),  # ukuran input model
    "batch_size"     : 32,          # kurangi ke 16 jika GPU OOM
    "epochs_warmup"  : 10,
    "epochs_finetune": 40,          # tambah jika akurasi masih rendah
    "unfreeze_layers": 60,          # tambah jika underfitting
    "dropout_rate"   : 0.4,
    "label_smoothing": 0.1,
    "use_mixed_precision": True,    # set False jika GPU lama
}
```

---

## 🎯 Tips Meningkatkan Akurasi

| Problem | Solusi |
|---------|--------|
| Akurasi < 80% | Tambah data, tambah epoch fine-tune |
| Overfitting (val acc turun) | Naikkan dropout_rate ke 0.5, tambah L2 |
| Training lambat | Kurangi batch_size, aktifkan mixed_precision |
| Model besar | Pakai EfficientNetV2B0 (lebih kecil dari V2S) |
| Akurasi kelas "Sad" rendah | Cek jumlah data sad, gunakan class_weight |

---

## 📊 Ekspektasi Akurasi

| Dataset | Kelas | Ekspektasi Akurasi |
|---------|-------|-------------------|
| FER2013 (7 kelas) | 7 | ~65-75% |
| FER2013 (4 kelas kita) | 4 | **~75-85%** |
| RAF-DB (4 kelas kita) | 4 | **~85-90%** |

> ℹ️ Akurasi meningkat saat menggunakan 4 kelas karena kelas sulit (disgust, fear, surprise) dihapus.

---

## 🔧 Arsitektur Model

```
Input (224×224×3)
    ↓
EfficientNetV2S (ImageNet pretrained)
    ↓
GlobalAveragePooling2D
    ↓
BatchNorm → Dense(512, ReLU) → Dropout(0.4)
    ↓
BatchNorm → Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(4, Softmax)  ← [Angry, Happy, Neutral, Sad]
```

**Optimizer:** AdamW + weight_decay + gradient clipping
**Loss:** Categorical Crossentropy + Label Smoothing (0.1)
**Class Weights:** Otomatis dihitung dari distribusi data

---

## 📦 Requirements

```
tensorflow>=2.12
opencv-python>=4.7
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
pillow>=9.4
```