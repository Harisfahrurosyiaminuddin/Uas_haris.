# Install library (jika di Colab)
!pip install tensorflow pandas scikit-learn

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Dataset minimal 20 berita (10 hoaks, 10 asli)
data = {
    'text': [
        "Vaksin menyebabkan autisme",  # hoaks
        "Pemerintah resmi menetapkan cuti bersama",
        "Covid-19 tidak ada, hanya konspirasi",  # hoaks
        "Bumi itu datar menurut penelitian baru",  # hoaks
        "Jokowi resmikan jalan tol baru",
        "Minum air lemon bisa sembuhkan kanker",  # hoaks
        "Polri umumkan operasi ketupat",
        "Alien ditemukan di hutan Amazon",  # hoaks
        "Gempa bumi akan terjadi besok",  # hoaks
        "Kemenkes umumkan penurunan kasus flu",
        "UFO muncul di atas gedung DPR",  # hoaks
        "Menteri keuangan paparkan RAPBN 2025",
        "Mie instan mengandung lilin",  # hoaks
        "Kemenkes tambah stok vaksin gratis",
        "Manusia berasal dari bangsa reptil",  # hoaks
        "Pemerintah larang ekspor beras",
        "5G penyebab virus corona",  # hoaks
        "Pemilu 2024 berlangsung aman",
        "Chip tertanam di vaksin Covid",  # hoaks
        "BMKG: Cuaca ekstrem diperkirakan minggu ini"
    ],
    'label': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0,
              1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Hoaks, 0 = Asli
}

df = pd.DataFrame(data)

# Tokenisasi teks
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, padding='post', maxlen=20)

# Data dan label
X = padded
y = np.array(df['label'])

# Split training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model CNN
model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=20),
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Training
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)

# Evaluasi
loss, acc = model.evaluate(X_test, y_test)
print(f"\nAkurasi: {acc * 100:.2f}%")
