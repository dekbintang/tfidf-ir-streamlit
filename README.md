# 📚 Sistem Information Retrieval dengan TF-IDF

Aplikasi ini merupakan implementasi **Information Retrieval (IR)** menggunakan metode **TF-IDF (Term Frequency - Inverse Document Frequency)** dan **Cosine Similarity** untuk mencari dokumen yang paling relevan berdasarkan query pengguna.

Aplikasi dibangun menggunakan:

* Streamlit (antarmuka web interaktif)
* Pandas (pengolahan data)
* PySastrawi (preprocessing Bahasa Indonesia)

---

## 🚀 Fitur Utama

* 📄 Menampilkan koleksi dokumen
* ⚙️ Preprocessing teks:

  * Tokenisasi
  * Stopword Removal
  * Stemming
* 📊 Perhitungan Term Frequency (TF)
* 📈 Perhitungan Inverse Document Frequency (IDF)
* 🔢 Perhitungan TF-IDF
* 🔍 Pencarian dokumen menggunakan Cosine Similarity
* 🏆 Ranking dokumen berdasarkan tingkat relevansi

---

## 🧠 Alur Sistem

```
Teks → Tokenisasi → Stopword Removal → Stemming → TF → IDF → TF-IDF → Cosine Similarity → Ranking
```

---

## 🗂️ Struktur Project

```
STKI3/
│── app.py              # Antarmuka utama (Streamlit)
│── documents.py        # Dataset dokumen
│── index_builder.py    # Membangun indeks TF-IDF
│── preprocessing.py    # Tokenisasi, stopword, stemming
│── stopwords.py        # Stopword dari Sastrawi
│── stemmer.py          # Stemming (Sastrawi)
│── tfidf.py            # TF, IDF, TF-IDF, Cosine Similarity
│── requirements.txt    # Dependency project
```

---

## ⚙️ Instalasi & Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone https://github.com/username/tfidf-ir-streamlit.git
cd tfidf-ir-streamlit
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan di browser secara otomatis.

---

## 📦 Dependencies

```txt
streamlit
pandas
PySastrawi
```

---

## 📊 Metode yang Digunakan

### 1. Term Frequency (TF)

Mengukur seberapa sering suatu kata muncul dalam dokumen.

```
TF(t,d) = jumlah kemunculan term / total kata dalam dokumen
```

### 2. Inverse Document Frequency (IDF)

Mengukur seberapa penting suatu kata dalam seluruh dokumen.

```
IDF(t) = log10(N / DF(t))
```

### 3. TF-IDF

Menggabungkan TF dan IDF untuk menentukan bobot kata.

```
TF-IDF = TF × IDF
```

### 4. Cosine Similarity

Digunakan untuk mengukur kemiripan antara query dan dokumen.
Nilai berkisar antara:

* 0 → tidak relevan
* 1 → sangat relevan

---

## 🔍 Cara Kerja Pencarian

1. User memasukkan query
2. Query diproses:

   * Tokenisasi
   * Stopword removal
   * Stemming
3. Hitung TF-IDF query
4. Hitung Cosine Similarity dengan setiap dokumen
5. Urutkan dokumen berdasarkan skor tertinggi

---

## 📌 Contoh Penggunaan

**Input:**

```
kesehatan tubuh olahraga
```

**Output:**

* Sistem menampilkan dokumen terkait kesehatan dan olahraga
* Disertai skor relevansi

---

## 🎯 Tujuan Proyek

Proyek ini dibuat untuk:

* Memahami konsep Information Retrieval
* Mengimplementasikan TF-IDF secara manual
* Mengolah teks Bahasa Indonesia dengan NLP
* Membangun aplikasi sederhana berbasis Streamlit

---

## 👤 Author

Nama: [Isi Nama Kamu]
Mata Kuliah: Sistem Temu Kembali Informasi (STKI)

---

## 📄 Lisensi

Proyek ini digunakan untuk keperluan pembelajaran dan pengembangan akademik.
