import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import joblib

load_dotenv()  # Load environment variables from .env file

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from unidecode import unidecode
import urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
import string
import requests # Impor requests untuk panggilan API
import random
from sentence_transformers import SentenceTransformer
import torch

# Instantiate SentenceTransformer model globally with explicit CPU device to avoid meta tensor error
# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Rekomendasi Destinasi Wisata Indonesia",
    page_icon="✈",
    layout="centered",
    initial_sidebar_state="auto"
)

try:
    import os
    os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"
    from sentence_transformers import SentenceTransformer
    sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
except Exception as e:
    sentence_transformer_model = None

# --- Gaya CSS untuk Tampilan Modern ---
st.markdown("""
<style>
.main-header {
    font-size: 3em;
    color: #2F80ED;
    text-align: center;
    margin-bottom: 30px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.subheader {
    font-size: 1.8em;
    color: #333;
    margin-top: 40px;
    margin-bottom: 100px; /* Ini yang menciptakan jarak setelah header */
    border-bottom: 2px solid #2F80ED;
    padding-bottom: 5px;
}
.stButton>button {
    background-color: #28a745;
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s, transform 0.2s;
}
.stButton>button:hover {
    background-color: #218838;
    color: black;
    transform: translateY(-2px);
}
.stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>select {
    border-radius: 8px;
    border: 1px solid #ccc;
    padding: 10px;
}
.destination-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 25px;
    display: flex;
    align-items: flex-start;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    background-color: #ffffff;
    transition: transform 0.2s;
    cursor: pointer; /* Agar terlihat bisa diklik */
}
.destination-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.12); /* Efek hover lebih menonjol */
}
.destination-card img {
    width: 200px;
    height: 130px;
    object-fit: cover;
    margin-right: 25px;
    border-radius: 8px;
}
.destination-details h4 {
    color: #2F80ED;
    font-size: 1.4em;
    margin-top: 0;
    margin-bottom: 8px;
}
.destination-details p {
    margin-bottom: 6px;
    color: #555;
    font-size: 0.95em;
}
.info-box {
    background-color: #e6f7ff;
    border-left: 5px solid #2F80ED;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 8px;
    color: #333;
}
.warning-box {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 8px;
    color: #856404;
}
.error-box {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 8px;
    color: #721c24;
}
.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 20px;
}
.gallery-grid img {
    width: 100%;
    height: 150px; /* Ukuran gambar di galeri */
    object-fit: cover;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# API Key untuk Pexels (Pastikan Anda menyetel variabel lingkungan ini)
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")

# Cache sederhana dalam memori untuk URL gambar per kueri (untuk satu gambar utama)
image_cache = {}
# Cache untuk kumpulan gambar (untuk halaman detail)
multi_image_cache = {}
used_images = set() # untuk menghindari duplikasi gambar utama di list

def get_pexels_images(queries, per_page=1, return_list=False):
    """
    Menerima daftar kueri, mencoba setiap kueri sampai gambar ditemukan.
    Menge-cache hasil.
    Jika return_list=True, akan mengembalikan daftar URL gambar.
    """
    api_key = PEXELS_API_KEY
    if not api_key:
        st.error("PEXELS_API_KEY tidak ditemukan. Mohon setel variabel lingkungan PEXELS_API_KEY.")
        return [] if return_list else None

    headers = {
        "Authorization": api_key
    }

    # Periksa cache yang sesuai
    current_cache = multi_image_cache if return_list else image_cache

    # Membuat string kunci unik untuk cache
    cache_key_prefix = "".join(queries) + f"{per_page}_{return_list}"
    
    if cache_key_prefix in current_cache:
        if return_list:
            return current_cache[cache_key_prefix]
        else:
            cached_urls = current_cache[cache_key_prefix]
            for url in cached_urls:
                if url not in used_images:
                    used_images.add(url)
                    return url
            # Jika semua gambar yang di-cache sudah digunakan untuk satu gambar, hapus cache dan ambil ulang
            del current_cache[cache_key_prefix]


    for query in queries:
        search_url = f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}&per_page={per_page}&orientation=landscape&size=medium"
        try:
            response = requests.get(search_url, headers=headers)
            if response.status_code == 401:
                st.error("Gagal mengakses Pexels API: Unauthorized (401). Periksa API key Anda.")
                return [] if return_list else None
            response.raise_for_status() # Munculkan exception untuk kesalahan HTTP
            data = response.json()
            photos = data.get('photos', [])

            if photos:
                found_urls = [photo['src']['medium'] for photo in photos]
                current_cache[cache_key_prefix] = found_urls # Cache semua URL yang ditemukan

                if return_list:
                    return found_urls
                else:
                    for url in found_urls:
                        if url not in used_images:
                            used_images.add(url)
                            return url
            else:
                st.warning(f"Tidak ada gambar ditemukan untuk kueri: {query}")
            # Jika tidak ada gambar yang ditemukan untuk kueri saat ini,
            # lanjutkan ke kueri berikutnya
        except requests.exceptions.RequestException as e:
            st.warning(f"Tidak dapat mengambil gambar dari Pexels untuk '{query}': {e}.")
            continue
    return [] if return_list else None

# Fungsi untuk pra-pemrosesan teks (disederhanakan dan konsisten)
def preprocess_text(text):
    if pd.isna(text): # Handle NaN values
        return ""
    text = unidecode(str(text)).lower()
    # Hapus angka dan tanda baca
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    # Hapus stop words
    custom_stop_words = list(ENGLISH_STOP_WORDS) + ['yang', 'dan', 'dengan', 'untuk', 'di', 'pada', 'adalah', 'ini', 'itu', 'atau'] # Tambahkan stop words Bahasa Indonesia jika perlu
    tokens = [word for word in tokens if word not in custom_stop_words]
    return ' '.join(tokens)

st.markdown("<h1 class='main-header'>Rekomendasi Destinasi Wisata Indonesia</h1>", unsafe_allow_html=True)
st.markdown("Selamat datang! Rencanakan liburan impian Anda dengan rekomendasi destinasi terbaik di seluruh Indonesia, disesuaikan dengan preferensi Anda.")

# --- Memuat Data dan Pra-pemrosesan (dengan caching) ---
@st.cache_data
def load_and_preprocess_data():
    """Memuat dan pra-memproses semua dataset yang diperlukan."""
    # Suppress info messages for loading and processing
    try:
        df_destinasi = pd.read_csv('destinasi-wisata-indonesia.csv')
        df_tourism_id = pd.read_csv('tourism_with_id.csv')
        df_rating = pd.read_csv('tourism_rating.csv')

        # Menggabungkan dataframe
        df_destinations_full = pd.merge(df_tourism_id, df_rating, on='Place_Id', how='left')
        df_destinations_full = pd.merge(df_destinations_full, df_destinasi[['Place_Id', 'Description', 'City']], on='Place_Id', how='left', suffixes=('', '_y'))
        df_destinations_full = df_destinations_full.loc[:,~df_destinations_full.columns.duplicated()]

        # Kolom wajib untuk pengelompokan
        required_cols_for_groupby = ['Place_Id', 'Place_Name', 'City', 'Category', 'Price', 'Description']
        for col in required_cols_for_groupby:
            if col not in df_destinations_full.columns:
                if col == 'Category':
                    df_destinations_full['Category'] = 'unknown'
                    st.warning("Kolom 'Category' tidak ditemukan, menggunakan nilai 'unknown' sebagai default.")
                elif col == 'Description':
                    df_destinations_full['Description'] = ''
                    st.warning("Kolom 'Description' tidak ditemukan, menggunakan string kosong sebagai default.")
                elif col == 'Price':
                    df_destinations_full['Price'] = 0
                    st.warning("Kolom 'Price' tidak ditemukan, menggunakan nilai 0 sebagai default.")
                else:
                    st.error(f"Kolom wajib '{col}' tidak ditemukan dalam data destinasi.")
                    return pd.DataFrame() # Mengembalikan DataFrame kosong jika kolom penting hilang

        # Mengelompokkan berdasarkan Place_Id dan menghitung rata-rata Rating
        df_destinations_full = df_destinations_full.groupby(['Place_Id', 'Place_Name', 'City', 'Category', 'Price', 'Description']) \
            .agg(Rating=('Rating', 'mean')) \
            .reset_index()

        # Mengubah 'Price' menjadi numerik dan mengisi nilai yang hilang
        df_destinations_full['Price_Numeric'] = df_destinations_full['Price'].replace({'Free': 0, 'Gratis': 0}).apply(pd.to_numeric, errors='coerce').fillna(0)
        df_destinations_full['Rating'] = df_destinations_full['Rating'].fillna(0)

        def categorize_destination_budget(price):
            """Mengategorikan destinasi berdasarkan harga."""
            if price < 30000: # Threshold untuk 'Hemat' agar lebih realistis untuk destinasi berbayar
                return 'Hemat'
            elif price <= 150000:
                return 'Standar'
            else:
                return 'Mewah'

        df_destinations_full['Destination_Budget_Category'] = df_destinations_full['Price_Numeric'].apply(categorize_destination_budget)

        def map_destination_category(category_name):
            """Memetakan kategori destinasi ke kategori yang lebih umum."""
            category_name = str(category_name).lower()
            if 'pantai' in category_name or 'pulau' in category_name or 'danau' in category_name or 'gunung' in category_name or 'air terjun' in category_name or 'taman nasional' in category_name or 'hutan' in category_name or 'bukit' in category_name or 'alam' in category_name or 'pegunungan' in category_name or 'sungai' in category_name:
                return 'alam'
            elif 'museum' in category_name or 'candi' in category_name or 'monumen' in category_name or 'sejarah' in category_name or 'kota tua' in category_name or 'situs' in category_name or 'prasasti' in category_name:
                return 'sejarah'
            elif 'budaya' in category_name or 'desa wisata' in category_name or 'galeri' in category_name or 'tradisional' in category_name or 'seni' in category_name or 'pertunjukan' in category_name:
                return 'budaya'
            elif 'kuliner' in category_name or 'restoran' in category_name or 'kafe' in category_name or 'makanan' in category_name or 'pasar makanan' in category_name or 'gastronomi' in category_name:
                return 'kuliner'
            elif 'belanja' in category_name or 'pusat perbelanjaan' in category_name or 'pasar' in category_name or 'mall' in category_name or 'oleh-oleh' in category_name or 'butik' in category_name:
                return 'belanja'
            elif 'petualangan' in category_name or 'outbound' in category_name or 'rafting' in category_name or 'trekking' in category_name or 'hiking' in category_name or 'diving' in category_name or 'snorkeling' in category_name or 'surfing' in category_name or 'olahraga air' in category_name:
                return 'petualangan'
            elif 'spa' in category_name or 'resort' in category_name or 'villa' in category_name or 'santai' in category_name or 'healing' in category_name or 'kesehatan' in category_name:
                return 'relaksasi'
            elif 'masjid' in category_name or 'gereja' in category_name or 'pura' in category_name or 'vihara' in category_name or 'kuil' in category_name or 'ziarah' in category_name or 'tempat ibadah' in category_name:
                return 'religi'
            elif 'anak' in category_name or 'keluarga' in category_name or 'taman bermain' in category_name or 'kebun binatang' in category_name or 'edukasi' in category_name or 'wahana' in category_name:
                return 'keluarga'
            elif 'modern' in category_name or 'gedung' in category_name or 'teknologi' in category_name or 'kota' in category_name or 'hiburan' in category_name or 'pusat kota' in category_name or 'instagramable' in category_name:
                return 'modern'
            else:
                return 'lain-lain'

        if 'Category' in df_destinations_full.columns:
            df_destinations_full['Mapped_Category'] = df_destinations_full['Category'].apply(map_destination_category)
        else:
            df_destinations_full['Mapped_Category'] = 'lain-lain'
            st.warning("Kolom 'Category' tidak ditemukan, 'Mapped_Category' diisi dengan 'lain-lain'.")

        # Suppress success message for data loaded
        # st.success("Data destinasi berhasil dimuat dan diproses.")
        return df_destinations_full

    except FileNotFoundError as e:
        st.error(f"Error: File CSV tidak ditemukan. Pastikan semua file CSV sudah ada di direktori yang sama dengan aplikasi Streamlit Anda. ({e})")
        st.stop() # Hentikan eksekusi jika file penting hilang
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
        st.stop()

# --- Melatih Model Pembelajaran Mesin (dengan caching sumber daya) ---
@st.cache_resource
def train_ml_model(df_destinations_full_local):
    """Melatih model klasifikasi dan TF-IDF vectorizer."""
    # Suppress info messages for training
    np.random.seed(42)
    num_samples = 1000

    preference_phrases = [
        'ingin liburan santai di pantai', 'mencari tempat makan enak di kota', 'suka wisata sejarah dan museum',
        'ingin belanja oleh-oleh murah', 'petualangan di gunung dan air terjun', 'healing di villa tenang',
        'suka tempat ramai dan banyak aktivitas', 'ingin melihat pemandangan indah', 'liburan bersama teman dengan budget terbatas',
        'mencari pengalaman budaya yang otentik', 'wisata religi dan tempat ibadah', 'liburan bersama anak-anak',
        'tempat instagramable dan modern', 'ingin mencoba makanan lokal', 'aktivitas outdoor yang menantang',
        'menikmati suasana pedesaan', 'berburu sunrise atau sunset', 'mencari ketenangan dan kedamaian',
        'liburan romantis bersama pasangan', 'tempat dengan pemandangan malam yang indah', '', 'tidak ada preferensi khusus',
    ]

    data = {
        'total_budget': np.random.randint(500000, 15000000, num_samples),
        'duration_days': np.random.randint(1, 15, num_samples),
        'activity_type': np.random.choice(['alam', 'kuliner', 'belanja', 'budaya', 'petualangan', 'relaksasi', 'sejarah', 'religi', 'keluarga', 'modern', 'lain-lain', 'unknown'], num_samples),
        'user_preference_text': np.random.choice(preference_phrases, size=num_samples),
        'num_adults': np.random.randint(1, 5, num_samples),
        'num_children': np.random.randint(0, 3, num_samples)
    }
    df_training = pd.DataFrame(data)

    def assign_holiday_category(row):
        """Menetapkan kategori liburan berdasarkan budget, durasi, dan jumlah orang."""
        budget = row['total_budget']
        duration = row['duration_days']
        num_total_people = row['num_adults'] + row['num_children']
        estimated_transport_acc_cost_ratio = 0.4
        remaining_budget_for_activities = budget * (1 - estimated_transport_acc_cost_ratio)

        if duration > 0 and num_total_people > 0:
            cost_per_person_per_day = remaining_budget_for_activities / (duration * num_total_people)
        else:
            cost_per_person_per_day = 0

        if cost_per_person_per_day < 300000:
            return 'Hemat'
        elif cost_per_person_per_day < 800000:
            return 'Standar'
        else:
            return 'Mewah'

    df_training['holiday_category'] = df_training.apply(assign_holiday_category, axis=1)

    X = df_training.drop('holiday_category', axis=1)
    y = df_training['holiday_category']

    numerical_features = ['total_budget', 'duration_days', 'num_adults', 'num_children']
    numerical_transformer = StandardScaler()

    categorical_features = ['activity_type']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    tfidf_vectorizer_for_model = TfidfVectorizer(stop_words='english', max_features=200)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop'
    )

    X_for_preprocessing = X.drop(columns=['user_preference_text'])
    X_processed_numeric_cat = preprocessor.fit_transform(X_for_preprocessing)
    
    # Pastikan preprocess_text diterapkan di sini
    X_tfidf_model = tfidf_vectorizer_for_model.fit_transform(X['user_preference_text'].fillna('').apply(preprocess_text))
    X_final = np.hstack((X_processed_numeric_cat, X_tfidf_model.toarray()))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_final, y) # Melatih pada data sintetis penuh untuk model produksi

    # Menyiapkan TF-IDF untuk kesamaan (similarity)
    corpus_for_similarity = pd.concat([
        df_training['user_preference_text'].fillna(''),
        df_destinations_full_local['Place_Name'].fillna(''),
        df_destinations_full_local['Description'].fillna('')
    ]).astype(str).tolist()

    if 'Mapped_Category' in df_destinations_full_local.columns:
        corpus_for_similarity.extend(df_destinations_full_local['Mapped_Category'].fillna('').astype(str).tolist())
    if 'City' in df_destinations_full_local.columns:
        corpus_for_similarity.extend(df_destinations_full_local['City'].fillna('').astype(str).tolist())

    # Pastikan preprocess_text diterapkan ke corpus untuk kesamaan
    corpus_for_similarity = [preprocess_text(text) for text in corpus_for_similarity]

    tfidf_vectorizer_for_similarity = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    tfidf_vectorizer_for_similarity.fit(corpus_for_similarity)

    df_destinations_full_local['Combined_Text_for_Similarity'] = \
        df_destinations_full_local['Place_Name'].fillna('') + ' ' + \
        df_destinations_full_local['Description'].fillna('') + ' ' + \
        df_destinations_full_local['Mapped_Category'].fillna('') + ' ' + \
        df_destinations_full_local['City'].fillna('')

    # Pra-pemrosesan teks untuk kesamaan
    df_destinations_full_local['Combined_Text_for_Similarity_Processed'] = df_destinations_full_local['Combined_Text_for_Similarity'].apply(preprocess_text)

    destination_tfidf_matrix = tfidf_vectorizer_for_similarity.transform(df_destinations_full_local['Combined_Text_for_Similarity_Processed'].astype(str))

    # Suppress success message for model trained
    # st.success("Model klasifikasi dan TF-IDF vectorizer berhasil dilatih.")
    return model, preprocessor, tfidf_vectorizer_for_model, tfidf_vectorizer_for_similarity, destination_tfidf_matrix

# Memuat data dan melatih model
df_destinations_full = load_and_preprocess_data()
model, preprocessor, tfidf_vectorizer_for_model, tfidf_vectorizer_for_similarity, destination_tfidf_matrix = train_ml_model(df_destinations_full.copy())

@st.cache_resource
def compute_destination_sentence_embeddings(df_destinations, _model_st):
    """Compute sentence transformer embeddings for destination combined text."""
    if _model_st is None:
        return None
    combined_texts = (df_destinations['Place_Name'].fillna('') + ' ' +
                      df_destinations['Description'].fillna('') + ' ' +
                      df_destinations['Mapped_Category'].fillna('') + ' ' +
                      df_destinations['City'].fillna('')).tolist()
    embeddings = _model_st.encode(combined_texts, convert_to_tensor=True)
    return embeddings

destination_sentence_embeddings = compute_destination_sentence_embeddings(df_destinations_full, sentence_transformer_model)

def get_recommendations(holiday_category, activity_type, user_preference_text, df_destinations,
                        city_filter=None, category_filter=None, rating_filter=0,
                        model_st=None, dest_tfidf_matrix=None, destination_sentence_embeddings=None, use_sentence_transformer=True,
                        max_recommendations=10):
    """
    Fungsi untuk mendapatkan rekomendasi destinasi berdasarkan kategori liburan, aktivitas, dan preferensi.
    """
    if df_destinations.empty:
        st.warning("Dataset destinasi kosong. Tidak dapat memberikan rekomendasi.")
        return pd.DataFrame()

    filtered_destinations = df_destinations[
        df_destinations['Destination_Budget_Category'] == holiday_category
    ].copy()

    # --- Logika Kesamaan Cosine untuk Preferensi Teks dengan Sentence Transformer --- 
    if user_preference_text and user_preference_text.strip() != "" and model_st:
        try:
            if use_sentence_transformer:
                # Use SentenceTransformer model for encoding, not the RandomForestClassifier
                if hasattr(model_st, 'encode'):
                    user_preference_embedding = model_st.encode([user_preference_text.strip()], convert_to_tensor=True)
                else:
                    st.warning("Model yang diberikan tidak memiliki metode 'encode'. Melanjutkan tanpa filter ini.")
                    return df_destinations
            else:
                # Fallback to TF-IDF vectorizer similarity (if needed)
                processed_user_preference_text = preprocess_text(user_preference_text.strip())
                user_preference_embedding = model_st.transform([processed_user_preference_text])

            # Pastikan indeks cocok dengan df_destinations_full asli untuk slicing
            positional_indices = df_destinations_full.index.get_indexer(filtered_destinations.index)
            # Filter out invalid indices (-1)
            valid_positional_indices = positional_indices[positional_indices != -1]
            filtered_destinations_valid = filtered_destinations.iloc[positional_indices != -1]

            if use_sentence_transformer:
                if destination_sentence_embeddings is None:
                    st.warning("Destination sentence embeddings tidak tersedia. Melanjutkan tanpa filter ini.")
                    return df_destinations
                relevant_destination_embeddings_tensor = destination_sentence_embeddings[valid_positional_indices]
            else:
                if dest_tfidf_matrix is None:
                    st.warning("TF-IDF matrix tidak tersedia. Melanjutkan tanpa filter ini.")
                    return df_destinations
                relevant_destination_embeddings = dest_tfidf_matrix[valid_positional_indices]
                relevant_destination_embeddings_tensor = torch.tensor(relevant_destination_embeddings.toarray(), device=user_preference_embedding.device)

            if relevant_destination_embeddings_tensor.shape[0] > 0:
                similarity_scores_filtered = torch.nn.functional.cosine_similarity(user_preference_embedding, relevant_destination_embeddings_tensor)
                similarity_scores_filtered = similarity_scores_filtered.cpu().numpy()
                temp_df = filtered_destinations_valid.copy()
                temp_df['Similarity_Score'] = similarity_scores_filtered

                SIMILARITY_THRESHOLD = 0.25 # Lowered threshold for better matching
                highly_similar_destinations = temp_df[temp_df['Similarity_Score'] >= SIMILARITY_THRESHOLD].sort_values(by='Similarity_Score', ascending=False)

                if not highly_similar_destinations.empty:
                    filtered_destinations = highly_similar_destinations
                else:
                    filtered_destinations['Similarity_Score'] = similarity_scores_filtered
                    filtered_destinations = filtered_destinations.sort_values(by='Similarity_Score', ascending=False)
            else:
                pass

        except Exception as e:
            st.warning(f"Error memproses preferensi teks dengan kesamaan cosine: {e}. Melanjutkan tanpa filter ini.")


    # Filter berdasarkan activity_type (Mapped_Category) jika tidak ada kecocokan preferensi teks yang kuat
    if not (user_preference_text and user_preference_text.strip() != "") and activity_type and activity_type != "Semua" and not filtered_destinations.empty:
        if 'Mapped_Category' in filtered_destinations.columns:
            filtered_destinations = filtered_destinations[
                filtered_destinations['Mapped_Category'].astype(str).str.contains(activity_type, case=False, na=False)
            ]
        elif 'Category' in filtered_destinations.columns: # Fallback
            filtered_destinations = filtered_destinations[
                filtered_destinations['Category'].astype(str).str.contains(activity_type, case=False, na=False)
            ]

    # Menerapkan filter tambahan jika disediakan
    if city_filter and city_filter != "Semua" and 'City' in filtered_destinations.columns and not filtered_destinations.empty:
        filtered_destinations = filtered_destinations[
            filtered_destinations['City'].astype(str).str.contains(city_filter, case=False, na=False)
        ]

    if category_filter and category_filter != "Semua" and not filtered_destinations.empty:
        if 'Mapped_Category' in filtered_destinations.columns:
            filtered_destinations = filtered_destinations[
                filtered_destinations['Mapped_Category'].astype(str).str.contains(category_filter, case=False, na=False)
            ]
        elif 'Category' in filtered_destinations.columns: # Fallback
            filtered_destinations = filtered_destinations[
                filtered_destinations['Category'].astype(str).str.contains(category_filter, case=False, na=False)
            ]

    if rating_filter > 0 and 'Rating' in filtered_destinations.columns and not filtered_destinations.empty:
        filtered_destinations = filtered_destinations[
            filtered_destinations['Rating'] >= rating_filter
        ]

    # Mengurutkan destinasi yang tersisa
    if not filtered_destinations.empty:
        # Prioritaskan pengurutan berdasarkan Similarity_Score jika ada, lalu Rating
        if 'Similarity_Score' in filtered_destinations.columns:
            filtered_destinations = filtered_destinations.sort_values(by=['Similarity_Score', 'Rating'], ascending=[False, False])
        elif 'Rating' in filtered_destinations.columns:
            filtered_destinations = filtered_destinations.sort_values(by='Rating', ascending=False)
        else:
            st.warning("Kolom 'Rating' atau 'Similarity_Score' tidak ditemukan untuk pengurutan. Mengabaikan pengurutan.")
        filtered_destinations = filtered_destinations.drop_duplicates(subset=['Place_Id'])
    else:
        return pd.DataFrame()

    # --- Logika Diversifikasi Selama Sampling ---
    final_recommendations = pd.DataFrame()
    if not filtered_destinations.empty:
        sampled_place_ids = []
        seen_categories = set()
        seen_cities = set()

        for index, row in filtered_destinations.iterrows():
            if len(sampled_place_ids) >= max_recommendations:
                break

            place_id = row['Place_Id']
            category_to_use = row['Mapped_Category'] if 'Mapped_Category' in row and pd.notna(row['Mapped_Category']) else (row['Category'] if 'Category' in row and pd.notna(row['Category']) else 'unknown')
            city = row['City'] if 'City' in row and pd.notna(row['City']) else 'unknown'

            if place_id not in sampled_place_ids:
                count_in_cat = sum(1 for rec_cat in final_recommendations['Mapped_Category'] if rec_cat == category_to_use) if 'Mapped_Category' in final_recommendations.columns else 0
                count_in_city = sum(1 for rec_city in final_recommendations['City'] if rec_city == city) if 'City' in final_recommendations.columns else 0

                # Relax diversification limits to increase variety
                # Coba batasi jumlah rekomendasi per kategori dan kota agar lebih bervariasi
                # Misalnya, maksimal 2-3 destinasi per kategori utama dan 2-3 per kota
                max_per_category = 3
                max_per_city = 3
                
                # Jika kategori atau kota belum terlalu banyak direkomendasikan
                if (count_in_cat < max_per_category) and (count_in_city < max_per_city):
                    sampled_place_ids.append(place_id)
                    final_recommendations = pd.concat([final_recommendations, pd.DataFrame([row])], ignore_index=True)
                    seen_categories.add(category_to_use)
                    seen_cities.add(city)
                elif len(final_recommendations) < 5: # Pastikan setidaknya 5 rekomendasi terlepas dari diversifikasi ketat
                    sampled_place_ids.append(place_id)
                    final_recommendations = pd.concat([final_recommendations, pd.DataFrame([row])], ignore_index=True)
                
        # Jika masih kurang dari max_recommendations rekomendasi, tambahkan yang tersisa tanpa mempertimbangkan diversifikasi lagi
        if len(final_recommendations) < max_recommendations:
            remaining_destinations = filtered_destinations[~filtered_destinations['Place_Id'].isin(sampled_place_ids)]
            remaining_to_add = min(max_recommendations - len(final_recommendations), len(remaining_destinations))
            if remaining_to_add > 0:
                final_recommendations = pd.concat([final_recommendations, remaining_destinations.head(remaining_to_add)], ignore_index=True)

        # Pastikan kolom 'Mapped_Category' ada
        if 'Mapped_Category' not in final_recommendations.columns and 'Category' in final_recommendations.columns:
            final_recommendations['Mapped_Category'] = final_recommendations['Category'].apply(lambda x: x if pd.notna(x) else 'unknown')
        elif 'Mapped_Category' not in final_recommendations.columns:
            final_recommendations['Mapped_Category'] = 'unknown'

        return final_recommendations.drop_duplicates(subset=['Place_Id'])

    return pd.DataFrame()


# Inisialisasi session state untuk navigasi
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'main_page' # Bisa 'main_page' atau 'detail_page'
if 'selected_destination' not in st.session_state:
    st.session_state['selected_destination'] = None


# Fungsi untuk menampilkan halaman detail destinasi
def display_destination_detail(destination_data):
    st.markdown(f"<h2 class='subheader'>{destination_data['Place_Name']}</h2>", unsafe_allow_html=True)

    # Add vertical space here
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True) # Adds 20px space

    # Tombol kembali
    if st.button("← Kembali ke Rekomendasi"):
        st.session_state['current_view'] = 'main_page'
        st.session_state['selected_destination'] = None
        st.rerun()
        return # Penting untuk keluar dari fungsi setelah rerun()

    st.markdown("---")

    col_main_img, col_details = st.columns([1, 2])

    with col_main_img:
        # Hasilkan kueri untuk mendapatkan gambar utama
        category_part = destination_data['Mapped_Category'] if 'Mapped_Category' in destination_data and pd.notna(destination_data['Mapped_Category']) else destination_data.get('Category', 'unknown')
        queries_main_image = [
            f"{destination_data['Place_Name']} {destination_data['City']} {category_part} Indonesia".strip(),
            f"{destination_data['Place_Name']} {destination_data['City']} Indonesia".strip(),
            f"{destination_data['Place_Name']} Indonesia".strip()
        ]
        main_image_url = get_pexels_images(queries_main_image, per_page=1, return_list=False)
        if not main_image_url:
            main_image_url = "https://via.placeholder.com/600x400?text=Gambar+Tidak+Tersedia"
        st.image(main_image_url, caption=destination_data['Place_Name'], use_container_width=True)

    with col_details:
        st.markdown(f"<p><strong>Lokasi:</strong> {destination_data['City']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Kategori:</strong> {destination_data['Mapped_Category'] if 'Mapped_Category' in destination_data and pd.notna(destination_data['Mapped_Category']) else destination_data.get('Category', 'N/A')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Rating:</strong> {destination_data['Rating']:.1f} / 5</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Perkiraan Biaya Masuk:</strong> Rp {destination_data['Price_Numeric']:,.0f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Kategori Anggaran Destinasi:</strong> <span style='font-weight: bold; color: green;'>{destination_data['Destination_Budget_Category']}</span></p>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<p><strong>Deskripsi Lengkap:</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p>{destination_data.get('Description', 'Tidak ada deskripsi lengkap untuk destinasi ini.')}</p>", unsafe_allow_html=True)

        # Link Google Maps
        search_query_maps = f"{destination_data['Place_Name']}, {destination_data['City']}, Indonesia"
        # Perbaiki URL Google Maps, harusnya https://www.google.com/maps/search/
        Maps_url = f"https://www.google.com/maps/search/{urllib.parse.quote(search_query_maps)}" 
        st.markdown(f"[📍 Lihat di Google Maps]({Maps_url})", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 class='subheader'>Galeri Foto</h3>", unsafe_allow_html=True)

    # Ambil kumpulan gambar untuk galeri (per_page bisa lebih tinggi)
    queries_gallery = [
        f"{destination_data['Place_Name']} {destination_data['City']} {category_part} Indonesia".strip(),
        f"{destination_data['Place_Name']} {category_part} Indonesia".strip(),
        f"{destination_data['City']} {category_part} Indonesia".strip(),
        f"{destination_data['Place_Name']} landscape Indonesia".strip()
    ]
    gallery_images = get_pexels_images(queries_gallery, per_page=6, return_list=True) # Ambil 6 gambar

    if gallery_images:
        st.markdown('<div class="gallery-grid">', unsafe_allow_html=True)
        # Menggunakan kolom Streamlit untuk layout grid yang lebih baik tanpa CSS native grid
        cols_gallery = st.columns(3) # Tampilkan 3 gambar per baris
        for i, img_url in enumerate(gallery_images):
            with cols_gallery[i % 3]:
                st.image(img_url, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Tidak ada gambar tambahan untuk galeri destinasi ini.")


# Logika tampilan utama berdasarkan session state
if st.session_state['current_view'] == 'detail_page' and st.session_state['selected_destination'] is not None:
    display_destination_detail(st.session_state['selected_destination'])
else: # Tampilkan halaman utama (input dan rekomendasi)
    st.markdown("---")
    st.markdown("<h3 class='subheader'>Input Informasi Liburan Anda:</h3>", unsafe_allow_html=True)
    with st.form("travel_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            total_budget = st.number_input('Total Budget (Rp):', min_value=100000, value=5000000, step=100000, format="%d")
            duration_days = st.slider('Lama Liburan (Hari):', min_value=1, max_value=30, value=5)
            num_adults = st.slider('Jumlah Dewasa:', min_value=1, max_value=10, value=2)
            num_recommendations = st.number_input('Jumlah Rekomendasi Destinasi:', min_value=1, max_value=50, value=10, step=1)
        with col2:
            activity_type_options = ['Semua', 'alam', 'kuliner', 'belanja', 'budaya', 'petualangan', 'relaksasi', 'sejarah', 'religi', 'keluarga', 'modern', 'lain-lain', 'unknown']
            activity_type = st.selectbox('Jenis Aktivitas Disukai (Opsional):', options=activity_type_options, index=0)
            num_children = st.slider('Jumlah Anak-anak:', min_value=0, max_value=10, value=0)
        user_preference_text = st.text_area(
            'Keterangan Keinginan Anda:',
            placeholder='Contoh: Ingin liburan santai di pantai, suka makanan pedas, mencari oleh-oleh unik.'
        )

        submit_button = st.form_submit_button("Dapatkan Rekomendasi")

        if submit_button:
            if total_budget <= 0 or duration_days <= 0 or (num_adults + num_children <= 0):
                st.error("Mohon lengkapi semua input dengan benar (Budget & Durasi harus angka positif, Jumlah Orang > 0).")
            else:
                user_input_df = pd.DataFrame([{
                    'total_budget': total_budget,
                    'duration_days': duration_days,
                    'activity_type': activity_type if activity_type != "Semua" else "lain-lain",
                    'user_preference_text': user_preference_text,
                    'num_adults': num_adults,
                    'num_children': num_children
                }])
                try:
                    # Pastikan user_preference_text diproses sebelum dikirim ke vectorizer_for_model
                    processed_user_preference_text_for_model = preprocess_text(user_input_df['user_preference_text'].iloc[0])
                    user_input_df['user_preference_text'] = processed_user_preference_text_for_model

                    user_input_processed_numeric_cat = preprocessor.transform(user_input_df.drop(columns=['user_preference_text']))
                    user_input_tfidf = tfidf_vectorizer_for_model.transform(user_input_df['user_preference_text'].fillna(''))
                    user_input_final = np.hstack((user_input_processed_numeric_cat, user_input_tfidf.toarray()))
                    predicted_category = model.predict(user_input_final)[0]

                    # Calculate cost per person per day
                    estimated_transport_acc_cost_ratio = 0.4
                    total_people = num_adults + num_children
                    if duration_days > 0 and total_people > 0:
                        cost_per_person_per_day = (total_budget * (1 - estimated_transport_acc_cost_ratio)) / (duration_days * total_people)
                    else:
                        cost_per_person_per_day = 0

                    # Determine category based on thresholds instead of ML prediction
                    if cost_per_person_per_day < 300000:
                        category_based_on_threshold = 'Hemat'
                    elif cost_per_person_per_day <= 800000:
                        category_based_on_threshold = 'Standar'
                    else:
                        category_based_on_threshold = 'Mewah'

                    st.session_state['predicted_category'] = category_based_on_threshold
                    st.session_state['last_activity_type'] = activity_type
                    st.session_state['last_user_preference_text'] = user_preference_text # Simpan teks asli untuk ditampilkan kembali
                    st.session_state['last_num_adults'] = num_adults
                    st.session_state['last_num_children'] = num_children
                    st.session_state['last_total_budget'] = total_budget
                    st.session_state['last_duration_days'] = duration_days

                    st.markdown("<h3 class='subheader'>Hasil Rencana Liburan:</h3>", unsafe_allow_html=True)

                    color_map = {
                        'Hemat': "#008957FF",     # Light Green
                        'Standar': '#FFD54F',   # Golden Yellow / Soft Orange
                        'Mewah': '#FF0000'      # Deep Red / Premium Red
                    }

                    color = color_map.get(category_based_on_threshold, 'black')

                    st.markdown(f"<div class='info-box'><p><b>Total Budget Anda:</b> Rp {total_budget:,.0f}</p>"
                                f"<p><b>Durasi Wisata:</b> {duration_days} hari</p>"
                                f"<p><b>Jumlah Orang:</b> {num_adults} Dewasa, {num_children} Anak-anak</p>"
                                f"<p><b>Perkiraan Biaya Per Hari Per Orang:</b> Rp {cost_per_person_per_day:,.0f}</p>"
                                f"<p><b>Kategori Liburan Prediksi:</b> <span style='color: {color}; font-weight: bold;'>{category_based_on_threshold}</span></p></div>", unsafe_allow_html=True)


                    initial_recommendations = get_recommendations(
                        category_based_on_threshold,
                        activity_type,
                        user_preference_text, # Preferensi teks asli diteruskan ke sini
                        df_destinations_full,
                        model_st=sentence_transformer_model, # Menggunakan SentenceTransformer model
                        dest_tfidf_matrix=destination_tfidf_matrix, # Menggunakan TF-IDF Matrix
                        destination_sentence_embeddings=destination_sentence_embeddings,
                        max_recommendations=num_recommendations
                    )

                    st.session_state['current_recommendations'] = initial_recommendations

                except Exception as e:
                    st.markdown(f"<div class='error-box'>Error saat mendapatkan rekomendasi: {e}. Mohon coba lagi.</div>", unsafe_allow_html=True)
                    st.session_state['current_recommendations'] = pd.DataFrame()

    # Bagian filter dan tampilan rekomendasi utama (jika sudah ada rekomendasi)
    if 'current_recommendations' in st.session_state and not st.session_state['current_recommendations'].empty:
        st.markdown("---")
        st.markdown("<h3 class='subheader'>Filter Rekomendasi (Opsional):</h3>", unsafe_allow_html=True)
        with st.form("filter_form"):
            all_cities_for_filter = ["Semua"] + sorted(df_destinations_full['City'].unique().tolist())
            all_categories_for_filter = ["Semua"] + sorted(df_destinations_full['Mapped_Category'].unique().tolist())

            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                city_filter = st.selectbox('Filter Kota:', options=all_cities_for_filter)
            with col_filter2:
                category_filter = st.selectbox('Filter Kategori:', options=all_categories_for_filter)
            with col_filter3:
                rating_filter = st.slider('Minimum Rating:', min_value=0.0, max_value=5.0, value=0.0, step=0.5)

            apply_filter_button = st.form_submit_button("Terapkan Filter")

        if apply_filter_button:
            predicted_category = st.session_state.get('predicted_category')
            activity_type_to_use = st.session_state.get('last_activity_type')
            user_preference_text_to_use = st.session_state.get('last_user_preference_text')

            if predicted_category is None:
                st.markdown("<div class='error-box'>Mohon tekan tombol 'Dapatkan Rekomendasi' terlebih dahulu untuk mendapatkan prediksi awal.</div>", unsafe_allow_html=True)
            else:
                filtered_recommendations = get_recommendations(
                    predicted_category,
                    activity_type_to_use,
                    user_preference_text_to_use,
                    df_destinations_full,
                    city_filter=city_filter,
                    category_filter=category_filter,
                    rating_filter=rating_filter,
                    model_st=sentence_transformer_model, # Menggunakan SentenceTransformer model
                    dest_tfidf_matrix=destination_tfidf_matrix, # Menggunakan TF-IDF Matrix
                    destination_sentence_embeddings=destination_sentence_embeddings,
                    max_recommendations=st.session_state.get('num_recommendations', 10)
                )
                st.session_state['current_recommendations'] = filtered_recommendations
                

        st.markdown("<h3 class='subheader'>Destinasi Rekomendasi Anda:</h3>", unsafe_allow_html=True)
        # Tambahkan spasi vertikal ekstra secara spesifik di sini
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True) # Sesuaikan 20px sesuai kebutuhan

        if not st.session_state['current_recommendations'].empty:
            for index, row in st.session_state['current_recommendations'].iterrows():
                category_part = ''
                if 'Mapped_Category' in row and pd.notna(row['Mapped_Category']):
                    category_part = row['Mapped_Category']
                elif 'Category' in row and pd.notna(row['Category']):
                    category_part = row['Category']

                queries = []
                queries.append(f"{row['Place_Name']} {row['City']} {category_part} Indonesia".strip())
                queries.append(f"{row['Place_Name']} {row['City']} Indonesia".strip())
                queries.append(f"{row['Place_Name']} Indonesia".strip())
                queries = list(dict.fromkeys(queries))

                # Ambil hanya satu gambar untuk tampilan kartu utama
                image_url = get_pexels_images(queries, per_page=1, return_list=False)
                if not image_url:
                    image_url = "https://via.placeholder.com/200x130?text=Gambar+Tidak+Tersedia"

                # Gunakan st.columns untuk layout dan st.button untuk klik
                col_img, col_text, col_button = st.columns([2, 5, 2])
                with col_img:
                    st.image(image_url, use_container_width=True)
                with col_text:
                    st.markdown(f"{row['Place_Name']}", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 0.9em; margin-bottom: 0;'>Lokasi: {row['City']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 0.9em; margin-bottom: 0;'>Rating: {row['Rating']:.1f} / 5</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 0.9em; color: green; font-weight: bold;'>{row['Destination_Budget_Category']}</p>", unsafe_allow_html=True)
                with col_button:
                    # Pastikan kita menyimpan seluruh baris data ke session_state untuk detail page
                    if st.button("Lihat Detail", key=f"detail_{row['Place_Id']}"):
                        st.session_state['selected_destination'] = row.to_dict() # Simpan seluruh baris sebagai dict
                        st.session_state['current_view'] = 'detail_page'
                        st.rerun()
            st.markdown("---") # Garis pemisah setelah semua kartu

        else:
            st.markdown("<div class='warning-box'>Tidak ada rekomendasi yang sesuai dengan kriteria dan filter yang Anda berikan. Coba longgarkan beberapa filter (misalnya, ganti kota atau kategori, atau turunkan rating minimum) untuk melihat lebih banyak pilihan.</div>", unsafe_allow_html=True)