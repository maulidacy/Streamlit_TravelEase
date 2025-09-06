# TravelEase ✈️

Selamat datang di **TravelEase**, aplikasi travel assistant modern yang dirancang untuk membantu Anda menemukan destinasi wisata terbaik di Indonesia sesuai dengan budget dan preferensi Anda. Dengan menggunakan kecerdasan buatan, TravelEase memberikan rekomendasi personal yang akurat dan menarik, membuat perencanaan liburan Anda lebih mudah dan menyenangkan!

**Live Demo:** https://travelease.streamlit.app/


## Fitur Utama

- **Homepage Interaktif**: Halaman utama yang ramah pengguna dengan navigasi intuitif dan tampilan modern.
- **Input Budget → Rekomendasi Otomatis**: Masukkan total budget, durasi liburan, jumlah orang, dan preferensi aktivitas. Sistem akan merekomendasikan destinasi wisata yang sesuai secara otomatis menggunakan model machine learning.
- **Kategori Wisata Lengkap**: Jelajahi destinasi berdasarkan kategori seperti alam, budaya, kuliner, belanja, dan lainnya. Termasuk opsi untuk destinasi terbaru, populer, dan sering dikunjungi.
- **UI Modern dengan Form Dinamis**: Form input muncul dengan animasi halus dari bawah layar saat tombol search ditekan, memberikan pengalaman pengguna yang smooth dan responsif.
- **Filter Lanjutan**: Filter berdasarkan kota, kategori, dan rating minimum untuk menyempurnakan rekomendasi.
- **Detail Destinasi Lengkap**: Klik "Lihat Detail" untuk melihat informasi lengkap, galeri foto, dan tautan Google Maps.
- **Galeri Foto Otomatis**: Integrasi dengan API Pixabay untuk menampilkan gambar destinasi yang relevan.
- **Load More**: Fitur "Lihat Lebih Banyak" untuk memuat rekomendasi tambahan tanpa reload halaman.

## Tech Stack

| Komponen          | Teknologi                  |
|-------------------|----------------------------|
| **Bahasa Pemrograman** | Python 3.10.9             |
| **Framework UI**  | Streamlit 1.24.1          |
| **Machine Learning** | Scikit-learn, Sentence Transformers, Torch 2.0.1 |
| **Database**      | CSV Files (destinasi-wisata-indonesia.csv, tourism_rating.csv, tourism_with_id.csv) |
| **Library Pendukung** | Pandas, NumPy, Requests, Unidecode, Python-dotenv |
| **API Eksternal** | Pixabay API untuk gambar   |
| **Deployment**    | Streamlit Cloud / Local    |

## Cara Instalasi & Menjalankan

Ikuti langkah-langkah berikut untuk menjalankan TravelEase di mesin Anda:

1. **Clone Repository**:
   ```bash
   git clone https://github.com/username/TravelEase.git
   cd TravelEase
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables** (Opsional, untuk API Pixabay):
   - Buat file `.env` di root direktori.
   - Tambahkan: `PIXABAY_API_KEY=your_api_key_here`
   - Jika tidak, aplikasi akan menggunakan fallback gambar default.

4. **Jalankan Aplikasi**:
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Akses di Browser**:
   - Buka `http://localhost:8501` di browser Anda.
   - Mulai jelajahi rekomendasi wisata!

**Catatan**: Pastikan semua file CSV (destinasi-wisata-indonesia.csv, tourism_rating.csv, tourism_with_id.csv) berada di direktori yang sama dengan streamlit_app.py.

## Struktur Folder Project

Berikut adalah struktur direktori proyek TravelEase:

```
TravelEase/
├── streamlit_app.py          # File utama aplikasi Streamlit
├── requirements.txt          # Daftar dependencies Python
├── runtime.txt               # Versi Python runtime
├── destinasi-wisata-indonesia.csv  # Data destinasi wisata
├── tourism_rating.csv        # Data rating destinasi
├── tourism_with_id.csv       # Data destinasi dengan ID
└── README.md                 # Dokumentasi proyek (file ini)
```

## Screenshots/Demo

### Homepage
![Homepage](https://via.placeholder.com/800x400?text=TravelEase+Homepage)  
*Tampilan homepage dengan form input dan rekomendasi awal.*

### Rekomendasi Destinasi
![Rekomendasi](https://via.placeholder.com/800x400?text=Destinasi+Rekomendasi)  
*Contoh rekomendasi destinasi berdasarkan input budget.*

### Detail Destinasi
![Detail](https://via.placeholder.com/800x400?text=Detail+Destinasi)  
*Halaman detail dengan galeri foto dan informasi lengkap.*

*(Screenshots akan diperbarui dengan gambar asli setelah deployment.)*

## Roadmap

Fitur-fitur yang direncanakan untuk pengembangan selanjutnya:

- **AI Rekomendasi Tingkat Lanjut**: Integrasi model AI lebih canggih untuk rekomendasi berbasis pola pengguna.
- **Integrasi API Hotel & Penerbangan**: Tambahkan opsi booking hotel dan tiket pesawat langsung dari aplikasi.
- **Mode Offline**: Fitur untuk menyimpan rekomendasi offline.
- **Multilingual Support**: Dukungan bahasa Indonesia dan Inggris.
- **Mobile App**: Versi aplikasi mobile menggunakan React Native atau Flutter.

## Kontribusi

Kami sangat terbuka untuk kontribusi dari komunitas! Jika Anda ingin berkontribusi:

1. Fork repository ini.
2. Buat branch baru untuk fitur Anda: `git checkout -b feature/AmazingFeature`.
3. Commit perubahan Anda: `git commit -m 'Add some AmazingFeature'`.
4. Push ke branch: `git push origin feature/AmazingFeature`.
5. Buat Pull Request.

Silakan baca [CONTRIBUTING.md](CONTRIBUTING.md) untuk panduan lebih detail. Ide, bug report, atau saran sangat diterima di [Issues](https://github.com/username/TravelEase/issues).

## Lisensi

Proyek ini dilisensikan di bawah **MIT License**. Lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.

---

**TravelEase** - Jadikan liburan Anda lebih mudah dan menyenangkan! 🌴✨

*Dibuat oleh Maulida Cahya*
