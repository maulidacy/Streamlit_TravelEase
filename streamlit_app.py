import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import requests # Import requests for API calls
import urllib.parse
import re
import string
from unidecode import unidecode
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- Load Environment Variables ---
load_dotenv()
# Mengubah variabel API key dari PEXELS_API_KEY menjadi PIXABAY_API_KEY
# Pastikan Anda telah mengatur variabel lingkungan PIXABAY_API_KEY di sistem Anda,
# atau ganti ini dengan kunci API Anda secara langsung jika Anda tidak menggunakan .env
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY")
# Atau, jika Anda ingin menyertakan langsung di kode (TIDAK DISARANKAN untuk produksi):
PIXABAY_API_KEY = "50961599-afae2344cae9faf87eae65a78"


# --- Default Images for Fallback ---
DEFAULT_IMAGES = {
    'alam': 'https://images.unsplash.com/photo-1506744038136-46273834b3fb',
    'budaya': 'https://images.unsplash.com/photo-1533856493584-0c6ca8ca9ce3',
    'kuliner': 'https://images.unsplash.com/photo-1510626176961-4b57d4fbad03',
    'belanja': 'https://images.unsplash.com/photo-1555529669-e69e7aa0ba9a',
    'lain-lain': 'https://images.unsplash.com/photo-1503917988258-f87a78e3c995'
}

def get_fallback_image(category):
    """Mengembalikan URL gambar default berdasarkan kategori destinasi"""
    return DEFAULT_IMAGES.get(category.lower() if category else 'lain-lain', DEFAULT_IMAGES['lain-lain'])

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Rekomendasi Destinasi Wisata Indonesia",
    page_icon="✈",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- CSS Styling for Modern Look ---
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
    margin-bottom: 10px;
    border-bottom: 2px solid #2F80ED;
    padding-bottom: 5px;
}
/* General button styling */
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

/* Specific styling for the button inside destination cards */
.destination-card .stButton>button {
    background-color: #007bff; /* Different color for card button */
    padding: 8px 15px; /* Smaller padding */
    font-size: 0.9em; /* Smaller font */
    display: block; /* Make it a block element to fill width if needed */
    width: fit-content; /* Adjust width to content */
    margin-top: 10px; /* Space from text */
}

.destination-card .stButton>button:hover {
    background-color: #0056b3;
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
    /* Removed cursor: pointer from card to make only button clickable */
}
.destination-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}
.destination-card img {
    width: 200px;
    height: 130px;
    object-fit: cover;
    margin-right: 25px;
    border-radius: 8px;
    border: 1px solid #e0e0e0; /* Added for improvement */
}
.destination-details {
    flex-grow: 1;
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
.detail-main-image {
    width: 100%;
    height: 300px;
    object-fit: cover;
    border-radius: 8px;
    margin-bottom: 20px;
}
.gallery-grid img {
    width: 100%;
    height: 150px;
    object-fit: cover;
    border-radius: 8px;
    transition: transform 0.3s;
}

.gallery-grid img:hover {
    transform: scale(1.03);
}
/* Hide Streamlit status widget that shows "Running ..." messages */
.stStatusWidget, .stSpinner {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# --- Global Model Initialization ---
@st.cache_resource
def load_sentence_transformer_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model SentenceTransformer: {e}. Rekomendasi berdasarkan teks mungkin terpengaruh.")
        return None
sentence_transformer_model = load_sentence_transformer_model()

# --- Pixabay API Image Fetching ---
# Cache for single images in list view (resets with new recommendations)
if 'image_cache_session' not in st.session_state:
    st.session_state['image_cache_session'] = {}
# Cache for multiple images in detail view (persistent across detail views)
if 'multi_image_cache' not in st.session_state:
    st.session_state['multi_image_cache'] = {}
# Tracks images used in the current list of recommendations to avoid immediate duplicates
if 'current_display_used_images' not in st.session_state:
    st.session_state['current_display_used_images'] = set()
# Initialize a global set to track used images
if 'used_images' not in st.session_state:
    st.session_state['used_images'] = set()

def get_pixabay_images(queries, per_page=3, return_list=False, cache_id=None):
    """
    Menerima daftar kueri, mencoba setiap kueri sampai gambar ditemukan dari Pixabay.
    Menge-cache hasil per sesi/per tampilan daftar.

    Perbaikan:
    - Query lebih spesifik dengan menambahkan "Indonesia"
    - Prioritas hasil yang lebih relevan
    - Meningkatkan per_page default ke 3 untuk hasil lebih banyak
    - Menyederhanakan dan mendiversifikasi query
    """
    api_key = PIXABAY_API_KEY # Menggunakan kunci API Pixabay
    if not api_key:
        st.error("Kunci API Pixabay tidak ditemukan. Mohon atur variabel lingkungan PIXABAY_API_KEY atau masukkan kunci Anda.")
        return [] if return_list else None

    # Pixabay tidak memerlukan header otorisasi seperti Pexels
    headers = {}

    # Gunakan hash yang lebih unik untuk cache key
    query_hash = hash(tuple(queries[:5]))  # Gunakan 5 query pertama untuk hash agar lebih unik
    cache_key = f"{cache_id}{query_hash}{per_page}_{return_list}"

    # Cek cache
    if return_list:  # Untuk multiple images (gallery)
        if cache_key in st.session_state['multi_image_cache']:
            return st.session_state['multi_image_cache'][cache_key]
    else:  # Untuk single image (main list)
        if cache_key in st.session_state['image_cache_session']:
            return st.session_state['image_cache_session'][cache_key]

    found_urls = []
    photos = []
    def clean_query(query):
        # Remove duplicate words but keep all instances of 'Indonesia' to avoid losing context
        words = query.lower().split()
        seen = set()
        cleaned_words = []
        for w in words:
            if w not in seen:
                cleaned_words.append(w)
                seen.add(w)
        return ' '.join(cleaned_words)

    for query in queries:
        cleaned_query = clean_query(query)
        # Remove forced addition of "Indonesia travel" to allow more flexible queries
        enhanced_query = cleaned_query
        params = {
            'key': api_key,
            'q': enhanced_query,  # Use cleaned query directly without urllib.quote_plus (urlencode will handle)
            'per_page': min(max(per_page, 3), 200),  # Ensure per_page is between 3 and 200
            'image_type': 'photo',
            'orientation': 'horizontal', # Mengatur orientasi untuk landscape
            'safesearch': 'true'
        }
        
        # Bangun URL dengan parameter
        search_url = "https://pixabay.com/api/?" + urllib.parse.urlencode(params)
        
        try:
            response = requests.get(search_url, headers=headers, timeout=5)
            # st.info(f"Pixabay API request URL: {search_url}")
            # st.info(f"Pixabay API response status: {response.status_code}")
            # Pixabay mengembalikan 400 untuk bad request atau parameter hilang, bukan 401 untuk API Key
            if response.status_code == 400:
                st.warning(f"Pixabay API Error (Status 400): {response.text}")
                continue # Coba query berikutnya jika ada
            
            response.raise_for_status()
            data = response.json()
            photos = data.get('hits', []) # Data gambar di Pixabay ada di 'hits'
            # st.info(f"Pixabay API returned {len(photos)} photos for query '{cleaned_query}'")

            if photos:
                if return_list:
                    current_query_urls = [photo['webformatURL'] for photo in photos]
                    # Filter out images already used
                    filtered_urls = [url for url in current_query_urls if url not in st.session_state['used_images']]
                    found_urls.extend(filtered_urls)
                    if len(found_urls) >= per_page:
                        break
                else:
                    # Single image logic
                    for photo in photos:
                        url = photo['webformatURL']
                        if url not in st.session_state['used_images']:
                            st.session_state['used_images'].add(url)
                            st.session_state['image_cache_session'][cache_key] = url
                            return url
        except Exception as e:
            st.warning(f"Pixabay API request failed: {e}")

    # Fallback jika tidak ada gambar yang memenuhi validasi atau ditemukan
    if return_list and found_urls:
        st.session_state['multi_image_cache'][cache_key] = found_urls
        return found_urls
    elif not return_list and photos:
        url = photos[0]['webformatURL']
        if url not in st.session_state['used_images']:
            st.session_state['used_images'].add(url)
            st.session_state['image_cache_session'][cache_key] = url
            return url
        
    return [] if return_list else None



# --- Text Preprocessing Function ---
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = unidecode(str(text)).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    'yang', 'dan', 'dengan', 'untuk', 'di', 'pada', 'adalah', 'ini', 'itu', 'atau',
    'ke', 'dari', 'sebagai', 'oleh', 'juga', 'karena', 'pada', 'saat', 'lebih', 'agar', 'bagi', 'dapat', 'akan', 'dalam']
    tokens = [word for word in tokens if word not in custom_stop_words]
    return ' '.join(tokens)

# --- Sentence Embeddings Computation ---
@st.cache_resource
def compute_destination_sentence_embeddings(df_destinations, _model_st):
    """Compute sentence transformer embeddings for destination combined text."""
    if _model_st is None:
        return None
    # st.info("Menghitung embeddings untuk destinasi... Ini mungkin membutuhkan waktu.") 
    
    place_name_series = df_destinations['Place_Name'].fillna('')
    description_series = df_destinations['Description'].fillna('')
    mapped_category_series = df_destinations['Mapped_Category'].fillna('')
    city_series = df_destinations['City'].fillna('')
    combined_texts = (place_name_series + ' ' + description_series + ' ' + mapped_category_series + ' ' + city_series).tolist()
    
    non_empty_texts_indices = [i for i, text in enumerate(combined_texts) if text.strip()]
    if not non_empty_texts_indices:
        st.warning("Tidak ada teks non-kosong untuk dienkode dalam data destinasi.")
        return None
    
    texts_to_encode = [combined_texts[i] for i in non_empty_texts_indices]
    
    batch_size = 32 
    all_embeddings = []
    for i in range(0, len(texts_to_encode), batch_size):
        batch_texts = texts_to_encode[i:i + batch_size]
        batch_embeddings = _model_st.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        all_embeddings.append(batch_embeddings.cpu()) 
    
    if not all_embeddings:
        return None
        
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    
    full_embeddings = torch.zeros((len(combined_texts), embeddings_tensor.shape[1]), device='cpu')
    for i, idx in enumerate(non_empty_texts_indices):
        full_embeddings[idx] = embeddings_tensor[i]
    
    return full_embeddings

# --- Location Extraction from User Text ---
def extract_location_from_text(user_text, available_cities):
    user_text_lower = user_text.lower()
    found_city = None
    longest_match_len = 0
    sorted_cities = sorted(available_cities, key=len, reverse=True)
    for city in sorted_cities:
        pattern = r'\b' + re.escape(city.lower()) + r'\b'
        if re.search(pattern, user_text_lower):
            if len(city) > longest_match_len:
                found_city = city
                longest_match_len = len(city)
    return found_city

# --- Streamlit UI Header ---
st.markdown("<h1 class='main-header'>Rekomendasi Destinasi Wisata Indonesia</h1>", unsafe_allow_html=True)
st.markdown("Selamat datang! Rencanakan liburan impian Anda dengan rekomendasi destinasi terbaik di seluruh Indonesia, disesuaikan dengan preferensi Anda.")

# --- Load and Preprocess Data (with caching) ---
@st.cache_data
def load_and_preprocess_data():
    """Memuat dan pra-memproses semua dataset yang diperlukan."""
    try:
        df_destinasi = pd.read_csv('destinasi-wisata-indonesia.csv')
        df_tourism_id = pd.read_csv('tourism_with_id.csv')
        df_rating = pd.read_csv('tourism_rating.csv')
        
        df_destinations_full = pd.merge(df_tourism_id, df_rating, on='Place_Id', how='left')
        df_destinations_full = pd.merge(df_destinations_full, df_destinasi[['Place_Id', 'Description', 'City']], on='Place_Id', how='left', suffixes=('', '_y'))
        
        df_destinations_full = df_destinations_full.loc[:,~df_destinations_full.columns.duplicated()]
        
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
                    return pd.DataFrame()
        
        df_destinations_full = df_destinations_full.groupby(['Place_Id', 'Place_Name', 'City', 'Category', 'Price', 'Description']) \
            .agg(Rating=('Rating', 'mean')) \
            .reset_index()
        
        df_destinations_full['Price_Numeric'] = df_destinations_full['Price'].replace({'Free': 0, 'Gratis': 0}).apply(pd.to_numeric, errors='coerce').fillna(0)
        df_destinations_full['Rating'] = df_destinations_full['Rating'].fillna(0)
        
        def categorize_destination_budget(price):
            if price < 30000:
                return 'Hemat'
            elif price <= 150000:
                return 'Standar'
            else:
                return 'Mewah'
        df_destinations_full['Destination_Budget_Category'] = df_destinations_full['Price_Numeric'].apply(categorize_destination_budget)
        
        def map_destination_category_to_five(category_name):
            category_name = str(category_name).lower()
            if 'kuliner' in category_name or 'restoran' in category_name or 'kafe' in category_name or 'makanan' in category_name or 'pasar makanan' in category_name or 'gastronomi' in category_name:
                return 'kuliner'
            elif 'belanja' in category_name or 'pusat perbelanjaan' in category_name or 'pasar' in category_name or 'mall' in category_name or 'oleh-oleh' in category_name or 'butik' in category_name:
                return 'belanja'
            elif 'budaya' in category_name or 'museum' in category_name or 'candi' in category_name or 'sejarah' in category_name or 'monumen' in category_name or 'desa wisata' in category_name or 'galeri' in category_name or 'tradisional' in category_name or 'seni' in category_name or 'pertunjukan' in category_name or 'kota tua' in category_name or 'situs' in category_name or 'prasasti' in category_name or 'religi' in category_name or 'masjid' in category_name or 'gereja' in category_name or 'pura' in category_name or 'vihara' in category_name or 'kuil' in category_name or 'ziarah' in category_name or 'tempat ibadah' in category_name:
                return 'budaya'
            elif 'pantai' in category_name or 'pulau' in category_name or 'danau' in category_name or 'gunung' in category_name or 'air terjun' in category_name or 'taman nasional' in category_name or 'hutan' in category_name or 'bukit' in category_name or 'alam' in category_name or 'pegunungan' in category_name or 'sungai' in category_name or 'petualangan' in category_name or 'outbound' in category_name or 'rafting' in category_name or 'trekking' in category_name or 'hiking' in category_name or 'diving' in category_name or 'snorkeling' in category_name or 'surfing' in category_name or 'olahraga air' in category_name:
                return 'alam'
            else:
                return 'lain-lain' 
        
        if 'Category' in df_destinations_full.columns:
            df_destinations_full['Mapped_Category'] = df_destinations_full['Category'].apply(map_destination_category_to_five)
        else:
            df_destinations_full['Mapped_Category'] = 'lain-lain'
            st.warning("Kolom 'Category' tidak ditemukan, 'Mapped_Category' diisi dengan 'lain-lain'.")
        
        return df_destinations_full
    except FileNotFoundError as e:
        st.error(f"Error: CSV file not found. Please ensure all CSV files are in the same directory as your Streamlit application. ({e})")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading or processing data: {e}")
        st.stop()

# --- Train Machine Learning Model (with resource caching) ---
@st.cache_resource
def train_ml_model(df_destinations_full_local):
    """Trains the classification model and TF-IDF vectorizer."""
    np.random.seed(42)
    num_samples = 1000
    preference_phrases = [
        'ingin liburan santai di alam terbuka', 'mencari tempat makan enak', 'suka belanja dan mencari oleh-oleh',
        'ingin pengalaman budaya yang otentik', 'petualangan di pegunungan', 'menikmati keindahan alam',
        'mencicipi kuliner khas daerah', 'berburu barang unik', 'belajar sejarah dan tradisi',
        'hiking dan eksplorasi alam', 'tidak ada preferensi khusus', 'liburan yang tenang di alam',
        'mencari makanan lokal', 'menjelajahi pasar tradisional', 'melihat situs bersejarah',
        'aktivitas outdoor', 'menikmati seni dan kerajinan', 'mencari restoran terbaik',
        'ingin berbelanja banyak', 'wisata budaya'
    ]
    data = {
        'total_budget': np.random.randint(500000, 15000000, num_samples),
        'duration_days': np.random.randint(1, 15, num_samples),
        'activity_type': np.random.choice(['alam', 'kuliner', 'belanja', 'budaya', 'lain-lain'], num_samples),
        'user_preference_text': np.random.choice(preference_phrases, size=num_samples),
        'num_adults': np.random.randint(1, 5, num_samples),
        'num_children': np.random.randint(0, 3, num_samples)
    }
    df_training = pd.DataFrame(data)

    def assign_holiday_category(row):
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
    
    X_tfidf_model = tfidf_vectorizer_for_model.fit_transform(X['user_preference_text'].fillna('').apply(preprocess_text))
    X_final = np.hstack((X_processed_numeric_cat, X_tfidf_model.toarray()))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_final, y)
    
    tfidf_vectorizer_for_similarity = None
    destination_tfidf_matrix = None
    
    if sentence_transformer_model is None:
        if df_destinations_full_local.empty:
            st.warning("df_destinations_full_local is empty, skipping TF-IDF similarity matrix creation during model training.")
        else:
            corpus_for_similarity = pd.concat([
                df_training['user_preference_text'].fillna(''),
                df_destinations_full_local['Place_Name'].fillna(''),
                df_destinations_full_local['Description'].fillna('')
            ]).astype(str).tolist()
            if 'Mapped_Category' in df_destinations_full_local.columns:
                corpus_for_similarity.extend(df_destinations_full_local['Mapped_Category'].fillna('').astype(str).tolist())
            if 'City' in df_destinations_full_local.columns:
                corpus_for_similarity.extend(df_destinations_full_local['City'].fillna('').astype(str).tolist())
            corpus_for_similarity = [preprocess_text(text) for text in corpus_for_similarity]
            
            tfidf_vectorizer_for_similarity = TfidfVectorizer(max_features=5000, ngram_range=(1,2)) 
            tfidf_vectorizer_for_similarity.fit(corpus_for_similarity)
            
            df_destinations_full_local['Combined_Text_for_Similarity'] = \
                df_destinations_full_local['Place_Name'].fillna('') + ' ' + \
                df_destinations_full_local['Description'].fillna('') + ' ' + \
                df_destinations_full_local['Mapped_Category'].fillna('') + ' ' + \
                df_destinations_full_local['City'].fillna('')
            df_destinations_full_local['Combined_Text_for_Similarity_Processed'] = df_destinations_full_local['Combined_Text_for_Similarity'].apply(preprocess_text)
            destination_tfidf_matrix = tfidf_vectorizer_for_similarity.transform(df_destinations_full_local['Combined_Text_for_Similarity_Processed'].astype(str))
            
    return model, preprocessor, tfidf_vectorizer_for_model, tfidf_vectorizer_for_similarity, destination_tfidf_matrix

# Initialize global variables to None
df_destinations_full = pd.DataFrame()
model = None
preprocessor = None
tfidf_vectorizer_for_model = None
tfidf_vectorizer_for_similarity = None
destination_tfidf_matrix = None
destination_sentence_embeddings = None

try:
    df_destinations_full = load_and_preprocess_data()
    if not df_destinations_full.empty:
        model, preprocessor, tfidf_vectorizer_for_model, tfidf_vectorizer_for_similarity, destination_tfidf_matrix = train_ml_model(df_destinations_full.copy())
        if sentence_transformer_model is not None:
            destination_sentence_embeddings = compute_destination_sentence_embeddings(df_destinations_full, sentence_transformer_model)
        else:
            st.warning("Model SentenceTransformer tidak dimuat, menggunakan TF-IDF untuk kesamaan teks.")
    else:
        st.error("Data destinasi kosong setelah pemrosesan awal. Tidak dapat melanjutkan dengan pelatihan model.")
except Exception as e:
    st.error(f"Error saat inisialisasi data atau model: {e}. Mohon periksa file CSV Anda dan integritas data.")

def get_recommendations(holiday_category, activity_type, user_preference_text, df_destinations,
                        city_filter=None, category_filter=None, rating_filter=0,
                        model_st=None, dest_tfidf_matrix=None, destination_sentence_embeddings=None,
                        max_recommendations_count=10): # Renamed for clarity
    """
    Fungsi untuk mendapatkan rekomendasi destinasi berdasarkan kategori liburan, aktivitas, dan preferensi.
    """
    if df_destinations.empty:
        st.warning("Dataset destinasi kosong. Tidak dapat memberikan rekomendasi.")
        return pd.DataFrame()

    filtered_destinations = df_destinations[
        df_destinations['Destination_Budget_Category'] == holiday_category
    ].copy()

    if filtered_destinations.empty:
        st.info(f"Tidak ada destinasi ditemukan untuk kategori anggaran '{holiday_category}'. Mohon coba sesuaikan anggaran Anda.")
        return pd.DataFrame()

    extracted_city_from_text = None
    if user_preference_text and df_destinations_full is not None and not df_destinations_full.empty:
        available_cities = [city.strip().lower() for city in df_destinations_full['City'].unique().tolist()]
        extracted_city_from_text = extract_location_from_text(user_preference_text.lower(), available_cities)
        if extracted_city_from_text:
            # st.info(f"Lokasi terdeteksi dari preferensi teks Anda: {extracted_city_from_text}. Filter otomatis diterapkan.")
            city_filter = extracted_city_from_text # Prioritize extracted city
    
    if city_filter and city_filter != "Semua" and 'City' in filtered_destinations.columns and not filtered_destinations.empty:
        city_filter_lower = city_filter.strip().lower()
        filtered_destinations = filtered_destinations[
            filtered_destinations['City'].astype(str).str.lower().str.strip() == city_filter_lower
        ]
        if filtered_destinations.empty:
            st.info(f"Tidak ada destinasi ditemukan setelah filter kota: '{city_filter}'.")
            return pd.DataFrame()

    if user_preference_text and user_preference_text.strip() != "":
        try:
            user_preference_embedding = None
            relevant_destination_embeddings_tensor = None
            
            if model_st is not None and destination_sentence_embeddings is not None:
                user_preference_embedding = model_st.encode([user_preference_text.strip()], convert_to_tensor=True, show_progress_bar=False)
                original_indices_in_full_df = df_destinations_full.index.get_indexer(filtered_destinations.index)
                valid_positional_indices = original_indices_in_full_df[original_indices_in_full_df != -1]
                if valid_positional_indices.size > 0:
                    relevant_destination_embeddings_tensor = destination_sentence_embeddings[valid_positional_indices]
            elif tfidf_vectorizer_for_similarity is not None and dest_tfidf_matrix is not None:
                processed_user_preference_text = preprocess_text(user_preference_text.strip())
                if processed_user_preference_text:
                    user_preference_embedding = tfidf_vectorizer_for_similarity.transform([processed_user_preference_text])
                    user_preference_embedding = torch.tensor(user_preference_embedding.toarray(), dtype=torch.float32, device='cpu')
                    
                    original_indices_in_full_df = df_destinations_full.index.get_indexer(filtered_destinations.index)
                    valid_positional_indices = original_indices_in_full_df[original_indices_in_full_df != -1]
                    if valid_positional_indices.size > 0:
                        relevant_destination_embeddings = dest_tfidf_matrix[valid_positional_indices]
                        if hasattr(relevant_destination_embeddings, 'toarray'):
                            relevant_destination_embeddings_tensor = torch.tensor(relevant_destination_embeddings.toarray(), dtype=torch.float32, device='cpu')
                        else:
                            relevant_destination_embeddings_tensor = torch.tensor(relevant_destination_embeddings, dtype=torch.float32, device='cpu')
                else:
                    st.warning("Teks preferensi pengguna yang diproses kosong. Melewatkan filter preferensi teks.")
            else:
                st.warning("Tidak ada model kesamaan teks yang tersedia (SentenceTransformer atau TF-IDF). Melewatkan filter preferensi teks.")
            
            if user_preference_embedding is not None and relevant_destination_embeddings_tensor is not None and relevant_destination_embeddings_tensor.shape[0] > 0:
                similarity_scores_filtered = torch.nn.functional.cosine_similarity(user_preference_embedding.to(relevant_destination_embeddings_tensor.device), relevant_destination_embeddings_tensor)
                similarity_scores_filtered = similarity_scores_filtered.cpu().numpy()
                
                temp_filtered_destinations = filtered_destinations.iloc[filtered_destinations.index.isin(df_destinations_full.index[valid_positional_indices])]
                temp_filtered_destinations['Similarity_Score'] = similarity_scores_filtered
                
                SIMILARITY_THRESHOLD = 0.25 
                highly_similar_destinations = temp_filtered_destinations[temp_filtered_destinations['Similarity_Score'] >= SIMILARITY_THRESHOLD].sort_values(by='Similarity_Score', ascending=False)
                
                if not highly_similar_destinations.empty:
                    filtered_destinations = highly_similar_destinations
                else:
                    filtered_destinations = temp_filtered_destinations.sort_values(by='Similarity_Score', ascending=False)
            else:
                st.warning("Embedding preferensi pengguna tidak dapat dibuat atau destinasi kosong untuk perhitungan kesamaan. Melewatkan filter preferensi teks.")
        except Exception as e:
            st.warning(f"Error memproses preferensi teks dengan kesamaan cosine: {e}. Melewatkan filter preferensi teks.")

    if activity_type and activity_type != "Semua" and not filtered_destinations.empty:
        filtered_destinations = filtered_destinations[
            filtered_destinations['Mapped_Category'].astype(str).str.contains(activity_type, case=False, na=False)
        ]
        
    if filtered_destinations.empty:
        st.info(f"Tidak ada destinasi ditemukan setelah filter jenis aktivitas: '{activity_type}'.")
        return pd.DataFrame()

    if category_filter and category_filter != "Semua" and not filtered_destinations.empty:
        filtered_destinations = filtered_destinations[
            filtered_destinations['Mapped_Category'].astype(str).str.contains(category_filter, case=False, na=False)
        ]
    if filtered_destinations.empty:
        st.info(f"Tidak ada destinasi ditemukan setelah filter kategori: '{category_filter}'.")
        return pd.DataFrame()

    if rating_filter > 0 and 'Rating' in filtered_destinations.columns and not filtered_destinations.empty:
        filtered_destinations = filtered_destinations[
            filtered_destinations['Rating'] >= rating_filter
        ]
    
    if filtered_destinations.empty:
        st.info(f"Tidak ada destinasi ditemukan setelah filter rating minimum: {rating_filter}.")
        return pd.DataFrame()

    if not filtered_destinations.empty:
        if 'Similarity_Score' in filtered_destinations.columns:
            # Sort by similarity, then by rating
            final_recommendations_unsampled = filtered_destinations.sort_values(by=['Similarity_Score', 'Rating'], ascending=[False, False])
        elif 'Rating' in filtered_destinations.columns:
            # Fallback to sort by rating only
            final_recommendations_unsampled = filtered_destinations.sort_values(by='Rating', ascending=False)
        else:
            # Default sort if no similarity or rating
            final_recommendations_unsampled = filtered_destinations
        
        final_recommendations_unsampled = final_recommendations_unsampled.drop_duplicates(subset=['Place_Id'])
    else:
        return pd.DataFrame()

    # --- Diversification Logic & Limiting ---
    # Apply global maximum limit AFTER all filtering and sorting
    GLOBAL_HARD_MAX_RECOMMENDATIONS = 100 # Increased hard limit to allow more recommendations

    if not final_recommendations_unsampled.empty:
        if len(final_recommendations_unsampled) <= max_recommendations_count:
            return final_recommendations_unsampled
        else:
            diversified_list = []
            sampled_place_ids = set()
            seen_categories = {}
            seen_cities = {}
            
            # Dynamically adjust diversity limits based on actual available categories
            all_available_categories = df_destinations['Mapped_Category'].unique().tolist()
            num_unique_categories = len([c for c in all_available_categories if c != 'lain-lain']) # Don't count 'lain-lain' for diversity
            
            # Ensure at least 1 destination per main category, but not too many
            max_per_category_limit = max(1, min(10, max_recommendations_count // max(1, num_unique_categories))) 
            max_per_city_limit = max(1, min(10, max_recommendations_count // 5)) 
            
            # Phase 1: Try to get a diverse set up to max_recommendations_count
            for index, row in final_recommendations_unsampled.iterrows():
                if len(diversified_list) >= max_recommendations_count:
                    break
                place_id = row['Place_Id']
                category_to_use = row['Mapped_Category'] if 'Mapped_Category' in row and pd.notna(row['Mapped_Category']) else 'lain-lain'
                city = row['City'] if 'City' in row and pd.notna(row['City']) else 'unknown'
                
                if place_id not in sampled_place_ids:
                    current_cat_count = seen_categories.get(category_to_use, 0)
                    current_city_count = seen_cities.get(city, 0)
                    
                    if (current_cat_count < max_per_category_limit and current_city_count < max_per_city_limit) or \
                       len(diversified_list) < (max_recommendations_count * 0.5): # Ensure initial set is somewhat diverse
                        diversified_list.append(row)
                        sampled_place_ids.add(place_id)
                        seen_categories[category_to_use] = current_cat_count + 1
                        seen_cities[city] = current_city_count + 1
            
            # Phase 2: If still not enough, fill up with remaining top recommendations (less strict on diversity)
            if len(diversified_list) < max_recommendations_count:
                remaining_destinations_sorted = final_recommendations_unsampled[~final_recommendations_unsampled['Place_Id'].isin(sampled_place_ids)]
                for index, row in remaining_destinations_sorted.iterrows():
                    if len(diversified_list) >= max_recommendations_count:
                        break
                    diversified_list.append(row)
                    sampled_place_ids.add(row['Place_Id'])
            
            # Final cut to the exact max_recommendations_count (or less if not enough)
            # And also apply the GLOBAL_HARD_MAX_RECOMMENDATIONS
            final_recommendations = pd.DataFrame(diversified_list).drop_duplicates(subset=['Place_Id']).head(max_recommendations_count)
            final_recommendations = final_recommendations.head(GLOBAL_HARD_MAX_RECOMMENDATIONS)
    
    return final_recommendations if not final_recommendations.empty else pd.DataFrame()

# --- Session State Initialization ---
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'main_page'
if 'selected_destination' not in st.session_state:
    st.session_state['selected_destination'] = None
# This will track the number of recommendations currently displayed in the main list
if 'displayed_recommendation_count' not in st.session_state:
    st.session_state['displayed_recommendation_count'] = 0
# This will store the full set of filtered recommendations (before cutting for display)
if 'full_filtered_recommendations' not in st.session_state:
    st.session_state['full_filtered_recommendations'] = pd.DataFrame()

# Callback function to handle the detail view
def handle_detail_click(place_id_clicked):
    st.session_state['selected_destination'] = df_destinations_full[df_destinations_full['Place_Id'] == place_id_clicked].iloc[0].to_dict()
    st.session_state['current_view'] = 'detail_page'

# Inject JavaScript to restore scroll position on page load
scroll_restore_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    const scrollPos = sessionStorage.getItem("scrollPosition");
    if (scrollPos) {
        window.scrollTo(0, parseInt(scrollPos));
        sessionStorage.removeItem("scrollPosition");
    }
});
</script>
"""
st.markdown(scroll_restore_js, unsafe_allow_html=True)

# --- Destination Detail Page Function ---
def display_destination_detail(destination_data):
    st.markdown(f"<h2 class='subheader'>{destination_data['Place_Name']}</h2>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    if st.button("← Kembali ke Rekomendasi"):
        # Inject JS to save scroll position before rerun
        save_scroll_js = """
        <script>
        sessionStorage.setItem("scrollPosition", window.scrollY);
        </script>
        """
        st.markdown(save_scroll_js, unsafe_allow_html=True)
        st.session_state['current_view'] = 'main_page'
        st.session_state['selected_destination'] = None
        # Clear images cache for main display when returning to main page
        st.session_state['image_cache_session'].clear()
        st.session_state['current_display_used_images'].clear()
        st.rerun()
        return
    st.markdown("---")
    col_main_img, col_details = st.columns([1, 2])
    with col_main_img:
        category_part = destination_data['Mapped_Category'] if 'Mapped_Category' in destination_data and pd.notna(destination_data['Mapped_Category']) else destination_data.get('Category', 'unknown')
        
        # Updated queries for main image (menggunakan get_pixabay_images)
        queries_main_image = [
            f"{destination_data['Place_Name']} {destination_data['City']} landmark",
            f"{destination_data['Place_Name']} tourist attraction",
            f"{destination_data['Mapped_Category']} {destination_data['City']}",
            f"{destination_data['Place_Name']} travel destination",
            f"{destination_data['City']} tourism spot",
            f"{destination_data['Place_Name']}",
            f"{destination_data['City']}"
        ]
        image_url = get_pixabay_images(queries_main_image, per_page=1, return_list=False, cache_id=destination_data['Place_Id']) 
        if not image_url:
            image_url = get_fallback_image(category_part) # Use fallback image
        st.image(image_url, caption=destination_data['Place_Name'], use_container_width=True)
    with col_details:
        st.markdown(f"<p><strong>Lokasi:</strong> {destination_data['City']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Kategori:</strong> {destination_data['Mapped_Category'] if 'Mapped_Category' in destination_data and pd.notna(destination_data['Mapped_Category']) else destination_data.get('Category', 'N/A')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Rating:</strong> {destination_data['Rating']:.1f} / 5</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Perkiraan Biaya Masuk:</strong> Rp {destination_data['Price_Numeric']:,.0f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Kategori Anggaran Destinasi:</strong> <span style='font-weight: bold; color: green;'>{destination_data['Destination_Budget_Category']}</span></p>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<p><strong>Deskripsi Lengkap:</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p>{destination_data.get('Description', 'Tidak ada deskripsi lengkap untuk destinasi ini.')}</p>", unsafe_allow_html=True)
        search_query_maps = f"{destination_data['Place_Name']}, {destination_data['City']}, Indonesia"
        Maps_url = f"http://maps.google.com/maps?q={urllib.parse.quote(search_query_maps)}"
        st.markdown(f"[📍 Lihat di Google Maps]({Maps_url})", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3 class='subheader'>Galeri Foto</h3>", unsafe_allow_html=True)
    
    # Updated queries for gallery (menggunakan get_pixabay_images)
    queries_gallery = [
        f"{destination_data['Place_Name']} {destination_data['City']} landscape",
        f"{destination_data['Place_Name']} tourist photos",
        f"{destination_data['Mapped_Category']} {destination_data['City']}",
        f"{destination_data['Place_Name']} travel photography",
        f"{destination_data['City']} tourism images",
        f"{destination_data['Place_Name']}",
        f"{destination_data['City']}"
    ]
    gallery_images = get_pixabay_images(queries_gallery, per_page=9, return_list=True, cache_id=f"detail_gallery_{destination_data['Place_Id']}")
    if gallery_images:
        st.markdown('<div class="gallery-grid">', unsafe_allow_html=True)
        cols_gallery = st.columns(3)
        for i, img_url in enumerate(gallery_images):
            with cols_gallery[i % 3]:
                st.image(img_url, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Tidak ada gambar tambahan untuk galeri destinasi ini.")

# --- Main Page Logic ---
if st.session_state['current_view'] == 'detail_page' and st.session_state['selected_destination'] is not None:
    display_destination_detail(st.session_state['selected_destination'])
else:
    st.markdown("---")
    st.markdown("<h3 class='subheader'>Input Informasi Liburan Anda:</h3>", unsafe_allow_html=True)
    with st.form("travel_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            total_budget = st.number_input('Total Budget (Rp):', min_value=100000, value=5000000, step=100000, format="%d")
            duration_days = st.slider('Lama Liburan (Hari):', min_value=1, max_value=30, value=5)
            num_adults = st.slider('Jumlah Dewasa:', min_value=1, max_value=10, value=2)
            # num_recommendations input removed as it's now dynamic
        with col2:
            activity_type_options = ['Semua', 'alam', 'budaya', 'kuliner', 'belanja']
            activity_type = st.selectbox('Jenis Aktivitas Disukai (Opsional):', options=activity_type_options, index=0)
            num_children = st.slider('Jumlah Anak-anak:', min_value=0, max_value=10, value=0)
        user_preference_text = st.text_area(
            'Keterangan Keinginan Anda:',
            placeholder='Contoh: Ingin liburan santai di pantai di Bali, suka makanan pedas di Bandung, mencari oleh-oleh unik di Jogja.'
        )
        submit_button = st.form_submit_button("Dapatkan Rekomendasi")

        if submit_button:
            if total_budget <= 0 or duration_days <= 0 or (num_adults + num_children <= 0):
                st.error("Mohon lengkapi semua input dengan benar (Budget & Durasi harus angka positif, Jumlah Orang > 0).")
            elif preprocessor is None or model is None or tfidf_vectorizer_for_model is None or df_destinations_full.empty:
                st.error("Model rekomendasi belum siap atau data destinasi kosong. Mohon periksa log di konsol untuk detailnya.")
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
                    processed_user_preference_text_for_model = preprocess_text(user_input_df['user_preference_text'].iloc[0])
                    user_input_df['user_preference_text'] = processed_user_preference_text_for_model
                    user_input_processed_numeric_cat = preprocessor.transform(user_input_df.drop(columns=['user_preference_text']))
                    user_input_tfidf = tfidf_vectorizer_for_model.transform(user_input_df['user_preference_text'].fillna(''))
                    user_input_final = np.hstack((user_input_processed_numeric_cat, user_input_tfidf.toarray()))
                    predicted_category = model.predict(user_input_final)[0]

                    estimated_transport_acc_cost_ratio = 0.4
                    total_people = num_adults + num_children
                    if duration_days > 0 and total_people > 0:
                        cost_per_person_per_day = (total_budget * (1 - estimated_transport_acc_cost_ratio)) / (duration_days * total_people)
                    else:
                        cost_per_person_per_day = 0
                    
                    # --- Dynamic max_recommendations_count based on budget, duration, people ---
                    if cost_per_person_per_day < 300000:
                        category_based_on_threshold = 'Hemat'
                        initial_display_count = 5 
                        load_more_step = 3
                    elif cost_per_person_per_day <= 800000:
                        category_based_on_threshold = 'Standar'
                        initial_display_count = 8
                        load_more_step = 4
                    else:
                        category_based_on_threshold = 'Mewah'
                        initial_display_count = 12
                        load_more_step = 5

                    # Adjust based on duration
                    if duration_days <= 3: initial_display_count = min(initial_display_count, 5)
                    elif duration_days >= 10: initial_display_count = max(initial_display_count, 10)

                    # Adjust based on people
                    if total_people >= 5: initial_display_count = max(initial_display_count, 7)
                    
                    # Ensure a reasonable minimum/maximum for initial display
                    initial_display_count = max(3, min(initial_display_count, 15)) 
                    
                    # Maximum total recommendations to ever show (hard limit for performance/UX)
                    GLOBAL_HARD_MAX_RECOMMENDATIONS_DISPLAY = 30 
                    
                    st.session_state['predicted_category'] = category_based_on_threshold
                    st.session_state['last_activity_type'] = activity_type
                    st.session_state['last_user_preference_text'] = user_preference_text
                    st.session_state['last_num_adults'] = num_adults
                    st.session_state['last_num_children'] = num_children
                    st.session_state['last_total_budget'] = total_budget
                    st.session_state['last_duration_days'] = duration_days
                    st.session_state['last_cost_per_person_per_day'] = cost_per_person_per_day
                    
                    # Store the step for 'Lihat Lebih Banyak'
                    st.session_state['load_more_step'] = load_more_step 
                    
                    # Clear session-specific caches for new recommendations
                    st.session_state['image_cache_session'].clear()
                    st.session_state['current_display_used_images'].clear()

                    # Fetch new recommendations on form submit
                    all_potential_recommendations = get_recommendations(
                        category_based_on_threshold,
                        activity_type,
                        user_preference_text,
                        df_destinations_full,
                        city_filter=None,
                        category_filter=None,
                        rating_filter=0,
                        model_st=sentence_transformer_model,
                        dest_tfidf_matrix=destination_tfidf_matrix,
                        destination_sentence_embeddings=destination_sentence_embeddings,
                        max_recommendations_count=GLOBAL_HARD_MAX_RECOMMENDATIONS_DISPLAY
                    )
                    st.session_state['full_filtered_recommendations'] = all_potential_recommendations
                    st.session_state['displayed_recommendation_count'] = initial_display_count

                except Exception as e:
                    st.markdown(f"<div class='error-box'>Error saat mendapatkan rekomendasi: {e}. Mohon coba lagi.</div>", unsafe_allow_html=True)
                    st.session_state['full_filtered_recommendations'] = pd.DataFrame()
                    st.session_state['displayed_recommendation_count'] = 0

        # Render "Hasil Rencana Liburan" section if session state variables exist
        if 'predicted_category' in st.session_state and 'last_total_budget' in st.session_state and 'last_duration_days' in st.session_state:
            st.markdown("<h3 class='subheader'>Hasil Rencana Liburan:</h3>", unsafe_allow_html=True)
            color_map = {
                'Hemat': "#008957FF",
                'Standar': '#FFD54F',
                'Mewah': '#FF0000'
            }
            color = color_map.get(st.session_state['predicted_category'], 'black')
            st.markdown(f"<div class='info-box'><p><b>Total Budget Anda:</b> Rp {st.session_state['last_total_budget']:,.0f}</p>"
                        f"<p><b>Durasi Wisata:</b> {st.session_state['last_duration_days']} hari</p>"
                        f"<p><b>Jumlah Orang:</b> {st.session_state.get('last_num_adults', 0)} Dewasa, {st.session_state.get('last_num_children', 0)} Anak-anak</p>"
                        f"<p><b>Perkiraan Biaya Per Hari Per Orang:</b> Rp {st.session_state.get('last_cost_per_person_per_day', 0):,.0f}</p>"
                        f"<p><b>Kategori Liburan Prediksi:</b> <span style='color: {color}; font-weight: bold;'>{st.session_state['predicted_category']}</span></p></div>", unsafe_allow_html=True)

    # Filter and display recommendations section (if recommendations exist)
    if not st.session_state['full_filtered_recommendations'].empty:
        st.markdown("---")
        st.markdown("<h3 class='subheader'>Filter Rekomendasi (Opsional):</h3>", unsafe_allow_html=True)
        with st.form("filter_form"):
            all_cities_for_filter = ["Semua"] + sorted(df_destinations_full['City'].unique().tolist())
            all_categories_for_filter = ['Semua', 'alam', 'budaya', 'kuliner', 'belanja', 'lain-lain']
            all_categories_for_filter = ["Semua"] + sorted([c for c in all_categories_for_filter if c != "Semua"])
            
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                city_filter_dropdown = st.selectbox('Filter Kota:', options=all_cities_for_filter)
            with col_filter2:
                category_filter_dropdown = st.selectbox('Filter Kategori:', options=all_categories_for_filter)
            with col_filter3:
                rating_filter_dropdown = st.slider('Minimum Rating:', min_value=0.0, max_value=5.0, value=0.0, step=0.5)
            apply_filter_button = st.form_submit_button("Terapkan Filter")

        if apply_filter_button:
            if preprocessor is None or model is None or tfidf_vectorizer_for_model is None or df_destinations_full.empty:
                st.error("Model rekomendasi belum siap atau data destinasi kosong untuk filter. Mohon tekan 'Dapatkan Rekomendasi' terlebih dahulu atau periksa log konsol.")
            else:
                predicted_category = st.session_state.get('predicted_category')
                activity_type_to_use = st.session_state.get('last_activity_type')
                user_preference_text_to_use = st.session_state.get('last_user_preference_text')
                
                if predicted_category is None:
                    st.markdown("<div class='error-box'>Mohon tekan tombol 'Dapatkan Rekomendasi' terlebih dahulu untuk mendapatkan prediksi awal.</div>", unsafe_allow_html=True)
                else:
                    st.session_state['image_cache_session'].clear()
                    st.session_state['current_display_used_images'].clear()

                    # Re-run get_recommendations with current filters and the full hard limit
                    GLOBAL_HARD_MAX_RECOMMENDATIONS_DISPLAY = 30 # Make sure this is consistent
                    filtered_recommendations = get_recommendations(
                        predicted_category,
                        activity_type_to_use,
                        user_preference_text_to_use,
                        df_destinations_full,
                        city_filter=city_filter_dropdown,
                        category_filter=category_filter_dropdown,
                        rating_filter=rating_filter_dropdown,
                        model_st=sentence_transformer_model,
                        dest_tfidf_matrix=destination_tfidf_matrix,
                        destination_sentence_embeddings=destination_sentence_embeddings,
                        max_recommendations_count=GLOBAL_HARD_MAX_RECOMMENDATIONS_DISPLAY # Refetch up to hard limit
                    )
                    st.session_state['full_filtered_recommendations'] = filtered_recommendations
                    
                    # Reset displayed count to initial based on budget, or to min(initial, len of new filtered)
                    initial_display_count_for_filter = 5 # Default initial display after filter
                    if st.session_state.get('predicted_category') == 'Hemat': initial_display_count_for_filter = 5
                    elif st.session_state.get('predicted_category') == 'Standar': initial_display_count_for_filter = 8
                    elif st.session_state.get('predicted_category') == 'Mewah': initial_display_count_for_filter = 12

                    st.session_state['displayed_recommendation_count'] = min(initial_display_count_for_filter, len(filtered_recommendations))
                    st.rerun() # Rerun to apply new filters and display

# Make the max recommendations display limit configurable via session state
if 'max_recommendations_display' not in st.session_state:
    st.session_state['max_recommendations_display'] = 30  # default limit

# Use the configurable limit in the main display logic
GLOBAL_HARD_MAX_RECOMMENDATIONS_DISPLAY = st.session_state['max_recommendations_display']

st.markdown("<h3 class='subheader'>Destinasi Rekomendasi Anda:</h3>", unsafe_allow_html=True)
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# Get the subset of recommendations to currently display
recommendations_to_show = st.session_state['full_filtered_recommendations'].head(st.session_state['displayed_recommendation_count'])
GLOBAL_HARD_MAX_RECOMMENDATIONS_DISPLAY = 30 # Ensure consistency

if not recommendations_to_show.empty:
    for index, row in recommendations_to_show.iterrows():
        category_part = row['Mapped_Category'] if 'Mapped_Category' in row and pd.notna(row['Mapped_Category']) else row.get('Category', 'unknown')
        
        # Updated queries for main card (menggunakan get_pixabay_images)
        queries = [
            f"{row['Place_Name']} {row['City']} landmark",
            f"{row['Place_Name']} tourist attraction",
            f"{row['Mapped_Category']} {row['City']}",
            f"{row['Place_Name']} travel destination",
            f"{row['City']} tourism spot",
            f"{row['Place_Name']}",
            f"{row['City']}"
        ]
        
        image_url = get_pixabay_images(queries, per_page=1, return_list=False, cache_id=row['Place_Id'])
        if not image_url:
            image_url = get_fallback_image(row['Mapped_Category'] if 'Mapped_Category' in row else 'lain-lain') # Use fallback image
        
        card_html = f"""
        <div class="destination-card">
            <img src="{image_url}" alt="{row['Place_Name']}">
            <div class="destination-details">
                <h4>{row['Place_Name']}</h4>
                <p>Lokasi: {row['City']}</p>
                <p>Kategori: {row['Mapped_Category'] if 'Mapped_Category' in row and pd.notna(row['Mapped_Category']) else row.get('Category', 'N/A')}</p>
                <p>Rating: {row['Rating']:.1f} / 5</p>
                <p style='color: green; font-weight: bold;'>{row['Destination_Budget_Category']}</p>
                <div>
                    </div>
            </div>
        </div>
        """
        # Use a container to place the HTML and then the Streamlit button within it
        with st.container():
            st.markdown(card_html, unsafe_allow_html=True)
            # Place the actual Streamlit button using a key for on_click
            # This button is rendered immediately after the card HTML in the Streamlit flow
            # which is why it visually appears below the card HTML initially, 
            # but we'll apply CSS to float it inside the card structure.
            st.button(
                "Lihat Detail", 
                key=f"detail_{row['Place_Id']}", 
                on_click=handle_detail_click, 
                args=(row['Place_Id'],)
            )
            # Custom CSS to move the button into the card layout correctly
            # This CSS targets the last button rendered before the markdown break
            st.markdown(f"""
            <style>
                div[data-testid="stVerticalBlock"] div:last-child > div > button[data-testid^="stButton-{f'detail_{row["Place_Id"]}'}"] {{
                    position: relative;
                    left: 230px; /* Adjust this value based on your image width + margin */
                    bottom: 50px; /* Move it up */
                    z-index: 1; /* Ensure it's above other elements if needed */
                }}
                /* Hide the default button styling that Streamlit adds outside the div */
                div[data-testid^="stHorizontalBlock"] div[data-testid^="stBlock"] > div[data-testid^="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] > div > button[data-testid^="stButton-detail_"] {{
                    display: block !important; /* Make sure it's visible after moving */
                }}
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("---") # Separator between cards

    # "Lihat Lebih Banyak" Button Logic
    if st.session_state['displayed_recommendation_count'] < len(st.session_state['full_filtered_recommendations']) and \
       st.session_state['displayed_recommendation_count'] < GLOBAL_HARD_MAX_RECOMMENDATIONS_DISPLAY:
        
        # Determine how many more to load
        load_step = st.session_state.get('load_more_step', 5) # Default to 5 if not set
        
        if st.button("Lihat Lebih Banyak", key="load_more_button_main"):
            st.session_state['displayed_recommendation_count'] += load_step
            # Ensure not to exceed available or global max
            st.session_state['displayed_recommendation_count'] = min(
                st.session_state['displayed_recommendation_count'], 
                len(st.session_state['full_filtered_recommendations']),
                GLOBAL_HARD_MAX_RECOMMENDATIONS_DISPLAY
            )
            st.rerun() 

if len(st.session_state['full_filtered_recommendations']) > 0:
    pass

else:
    pass