import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

# ---------- LOAD ENVIRONMENT (GROQ API KEY) ----------
# Changed to load the specific file name you have
load_dotenv('GROQ_API_KEY.env')

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🏠 Lahore Real Estate AI Analyst",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- BEAUTIFUL CUSTOM CSS (Gold & Black Theme) ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0d0d0d, #1a1a2e, #16213e);
    }
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: #1a1a2e;
        border: none;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px #f7971e;
    }
    .stSelectbox > div > div, .stNumberInput > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid #f7971e !important;
        border-radius: 8px !important;
        color: white !important;
    }
           /* ---------- STRONG CHAT FIX ---------- */
    
    /* All Chat Messages: Solid dark background, White text */
    [data-testid="stChatMessage"] {
        background-color: #16213e !important; /* Solid dark blue background */
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #FFFFFF !important;
    }

    /* FORCE all text inside user and assistant bubbles to be bright white */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* USER Message Bubble: Solid Gold background, Solid Black text */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: #f7971e !important; 
        border: none !important;
    }

    /* Force text inside USER bubble to black */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) span,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div {
        color: #000000 !important;
        opacity: 1 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="main-title">🏠 Lahore Real Estate AI Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">🤖 Ask about property prices in Lahore, or predict your own!</div>', unsafe_allow_html=True)

# ---------- 1. LOAD CLEANED DATA ----------
@st.cache_data
def load_data():
    try:
        # Fixed file name to match your folder
        df = pd.read_csv("lahore_data.csv")
        required_cols = ['price', 'location', 'area', 'bedrooms', 'bathrooms']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"❌ Missing columns in cleaned CSV: {missing}. Please re-run the cleaning script.")
            return None
        return df
    except FileNotFoundError:
        st.error("❌ 'lahore_data.csv' not found! Please run the cleaning script first.")
        return None

df = load_data()
if df is None:
    st.stop()

# ---------- 2. TRAIN PRICE PREDICTION MODEL (Cached) ----------
@st.cache_resource
def train_model(data):
    feature_cols = ['location', 'area', 'bedrooms', 'bathrooms', 'purpose', 'property_type']
    existing_feature_cols = [col for col in feature_cols if col in data.columns]
    
    if not existing_feature_cols:
        st.error("❌ No valid feature columns found for training!")
        return None, None, None, None, None
    
    X = data[existing_feature_cols]
    y = data['price']
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Pre-calculate stats for the chatbot
    stats = {
        'total': len(data),
        'avg_price': data['price'].mean(),
        'median_price': data['price'].median(),
        'min_price': data['price'].min(),
        'max_price': data['price'].max(),
        'avg_area': data['area'].mean() if 'area' in data.columns else 0,
        'location_counts': data['location'].value_counts().head(10).to_dict(),
        'avg_price_by_location': data.groupby('location')['price'].mean().sort_values(ascending=False).head(10).to_dict(),
        'feature_cols': existing_feature_cols,
        'head': data.head(10).to_string()
    }
    
    return model, stats, existing_feature_cols, r2, mae

model, stats, feature_cols, r2, mae = train_model(df)

if model is None:
    st.stop()

# ---------- 3. SIDEBAR (Dashboard + Prediction Form) ----------
with st.sidebar:
    st.markdown("## 📊 **Market Dashboard**")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏘️ Listings", f"{stats['total']:,}")
    with col2:
        st.metric("💰 Avg Price", f"PKR {stats['avg_price']:,.0f}")
    
    st.markdown("---")
    st.markdown("### 🏙️ Top Locations by Avg Price")
    for loc, price in list(stats['avg_price_by_location'].items())[:5]:
        st.markdown(f"- **{loc}**: PKR {price:,.0f}")
    
    st.markdown("---")
    st.markdown("## 🔮 **Predict Your Price**")
    st.markdown("Fill in the details below to estimate a property price.")
    
    # Build dynamic input form
    input_data = {}
    for col in feature_cols:
        if col in df.columns:
            if col == 'location':
                top_locations = df['location'].value_counts().head(20).index.tolist()
                input_data[col] = st.selectbox("📍 Location", top_locations)
            elif col == 'property_type':
                input_data[col] = st.selectbox("🏠 Property Type", sorted(df[col].dropna().unique()))
            elif col == 'purpose':
                input_data[col] = st.selectbox("📌 Purpose", sorted(df[col].dropna().unique()))
            elif col in ['bedrooms', 'bathrooms']:
                input_data[col] = st.number_input(f"🛏️ {col.capitalize()}", min_value=0, max_value=20, value=3)
            elif col == 'area':
                input_data[col] = st.number_input("📐 Area (Marla)", min_value=1.0, max_value=500.0, value=5.0)
    
    if st.button("🚀 Predict Price", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = df[col].mode()[0] if col in df.columns else 0
        input_df = input_df[feature_cols]
        
        try:
            pred = model.predict(input_df)[0]
            st.success(f"💰 Estimated Price: **PKR {pred:,.0f}**")
            st.caption(f"📊 Model Accuracy: R² = {r2:.2f} | Typical error: ±PKR {mae:,.0f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------- 4. GROQ AI SETUP ----------
try:
    from groq import Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        ai_available = False
        st.sidebar.warning("⚠️ GROQ_API_KEY not found in .env file.")
    else:
        client = Groq(api_key=groq_api_key)
        ai_available = True
except ImportError:
    ai_available = False
    st.sidebar.warning("⚠️ Groq library not installed. Run: pip install groq")

# ---------- 5. CHAT INTERFACE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- 6. CHAT LOGIC ----------
if prompt := st.chat_input("💬 Ask about Lahore real estate..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        if ai_available:
            # FIXED: The data_summary block is now perfectly structured
            data_summary = f"""
🏠 Lahore Real Estate Market Summary:
- Total Properties: {stats['total']}
- Average Price: PKR {stats['avg_price']:,.0f}
- Median Price: PKR {stats['median_price']:,.0f}
- Price Range: PKR {stats['min_price']:,.0f} to PKR {stats['max_price']:,.0f}
- Average Area: {stats['avg_area']:.1f} Marla

📍 Top 20 Locations by Average Price:
"""
            for loc, price in list(stats['avg_price_by_location'].items())[:20]:
                data_summary += f"- {loc}: PKR {price:,.0f}\n"

            data_summary += f"\n\n🗺️ Location Frequency (Most Listings):\n"
            for loc, count in list(stats['location_counts'].items())[:10]:
                data_summary += f"- {loc}: {count} properties\n"

            data_summary += f"\n\n📊 Sample Data (First 10 rows):\n{stats['head']}"

            system_prompt = f"""
You are an expert Real Estate Data Analyst for Lahore, Pakistan.
Answer the user's question STRICTLY based on the provided data below.

{data_summary}

Rules:
- If they ask for a specific location price, pull it from the "Top Locations" list.
- If they ask "What is the best location?" or "Where should I invest?", recommend based on the data provided.
- If they ask for properties under a certain price, tell them to use the prediction feature in the app.
- Be helpful, confident, and use a professional tone. Keep answers concise (max 150 words).
- If you don't know the exact answer based on the data, say "Based on the provided data, I suggest checking the Lahore average prices."
"""
            
            try:
                # FIXED: Changed MODEL_NAME to lowercase 'model'
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"❌ API Error: {e}. Please check your API key or model permissions."
        else:
            reply = f"📊 Offline Mode: You asked '{prompt}'. \n\nTo enable AI, create a `.env` file with `GROQ_API_KEY=your_key_here` and restart."
        
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------- FOOTER ----------
st.markdown("---")
st.caption(f"⚡ Model Accuracy: R² = {r2:.2f} | MAE = PKR {mae:,.0f} | Built with Streamlit + Groq | Data: Zameen.com Lahore")