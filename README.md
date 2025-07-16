# 🏨 Analisis Sentimen Ulasan Hotel – Naive Bayes
Proyek ini bertujuan untuk menganalisis sentimen ulasan pengguna hotel dari Traveloka dan mengklasifikasikannya ke dalam kategori positif, negatif, atau netral menggunakan algoritma Naive Bayes. Aplikasi dibangun dengan antarmuka web interaktif menggunakan Streamlit.

## 🧾 Deskripsi Proyek
Dalam industri pariwisata, ulasan pelanggan sangat penting untuk membangun reputasi hotel. Dengan melakukan analisis sentimen secara otomatis, pemilik hotel dapat memperoleh insight strategis secara cepat dan efisien.

## 🔍 Metodologi
1. Scraping Data
   - Mengambil data ulasan hotel dari Traveloka menggunakan Web Scraper
   - Fitur: Nama akun, rating, tanggal, dan isi ulasan
2. Labelling Sentimen
   - Menggunakan TextBlob + Google Translate API
   - Kategori: Positif, Netral, Negatif
3. Preprocessing
   - Cleaning
   - Case Folding
   - Tokenization
   - Stopword Removal
   - Stemming (Sastrawi)
4. Feature Engineering: 
   - TF-IDF: Representasi numerik teks
   - Chi-Square: Seleksi fitur (top 5000)
   - SMOTE: Penyeimbangan data
5. Modeling
   - Algoritma: Naive Bayes
   - Akurasi: 91%
6. Evaluasi: Confusion Matrix, Precision, Recall, F1-Score
