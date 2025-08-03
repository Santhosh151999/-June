import streamlit as st
import os
import pandas as pd
from transformers import pipeline

# --- THEME & CONFIG ---
st.set_page_config(page_title="Hashtags • #June", layout="wide")

st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(120deg, #171930 0%, #303fa2 70%, #412e56 100%);
    font-family: -apple-system,BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif;
    color: #e8f1fd;
}
.block-container {
    padding: 1rem 2rem;
}
.stTitle, .stHeader, .stSubheader, .stCaption, .stText, .stMarkdown, .stMetric {
    color: #e8f1fd !important;
}
.custom-table {
    width: 100%;
    border-collapse: collapse;
    border-radius: 16px;
    overflow: hidden;
    background-color: rgba(255, 255, 255, 0.04);
    box-shadow: 0 0 18px rgba(255,255,255,0.1);
}
.custom-table th, .custom-table td {
    padding: 12px 18px;
    text-align: left;
    color: #ebf7ff;
    font-weight: 500;
    border-bottom: 1px solid rgba(255,255,255,0.15);
}
.custom-table thead {
    background-color: rgba(255, 255, 255, 0.06);
    color: #87e8fb;
    font-weight: 700;
    font-size: 1.02rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🏷 Hashtag Explorer • #June")
st.caption("Filter trending tags across categories, regions, periods, and sentiment.")

# --- Load Data ---
def load_data(folder):
    if not os.path.exists(folder): return None
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    data = []
    for file in files:
        df = pd.read_csv(os.path.join(folder, file))
        df = df.rename(columns={
            "Hashtag": "tag",
            "Cleaned Tweet Count": "count",
            "Tweet Count": "count",
            "Cleaned Record Date": "date",
            "Scraped Date": "date",
            "Scraped Time": "time",
            "Hour Offset": "hour",
            "Rank": "rank",
            "Source": "region",
            "Period": "period"
        })
        for col in ['tag', 'count', 'date', 'time', 'hour', 'rank', 'period', 'region']:
            if col not in df.columns:
                df[col] = None
        data.append(df)
    return pd.concat(data, ignore_index=True) if data else None

df_now = load_data('Trend_Now')
df_ymw = load_data('Trend_YMW')

if df_now is not None and df_ymw is not None:
    df = pd.concat([df_now, df_ymw], ignore_index=True)
elif df_now is not None:
    df = df_now
elif df_ymw is not None:
    df = df_ymw
else:
    st.error("❌ No data found in either trend_now or trend_ymw.")
    st.stop()

# --- Clean up ---
df['tag'] = df['tag'].astype(str).str.strip().str.lower()
df['region'] = df['region'].astype(str).str.strip().str.lower().map({'india': 'India', 'world': 'World'}).fillna("Unknown")
df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
if 'date' in df: df['date'] = df['date'].astype(str)
if 'period' in df: df['period'] = df['period'].astype(str)

# --- Sentiment ---
st.info("🔍 Classifying hashtags for multilingual sentiment…")

@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

model = get_sentiment_model()

unique_tags = df['tag'].unique()
results = model(list(unique_tags), truncation=True)
sentiment_map = {}
for tag, res in zip(unique_tags, results):
    stars = ''.join(filter(str.isdigit, res['label']))
    if stars in ['4', '5']:
        sentiment_map[tag] = 'Positive'
    elif stars == '3':
        sentiment_map[tag] = 'Neutral'
    else:
        sentiment_map[tag] = 'Negative'
df['sentiment'] = df['tag'].map(sentiment_map)

# --- Filters ---
col1, col2 = st.columns(2)
with col1:
    search = st.text_input("🔎 Search hashtag by keyword:")
with col2:
    sentiment_filter = st.multiselect("🧠 Sentiment", ["Positive", "Neutral", "Negative"], default=[])

region_filter = st.multiselect("🌍 Region", sorted(df['region'].dropna().unique()), default=[])
min_count, max_count = df['count'].min(), df['count'].max()
count_range = st.slider("📊 Tweet Count", int(min_count), int(max_count), (int(min_count), int(max_count)))
period_filter = st.multiselect("⏱️ Period", sorted(df['period'].dropna().unique()), default=[])

# --- Apply filters ---
filtered = df.copy()
filtered = filtered[filtered['count'].between(count_range[0], count_range[1])]
if search:
    filtered = filtered[filtered['tag'].str.contains(search.strip(), case=False, na=False)]
if sentiment_filter:
    filtered = filtered[filtered['sentiment'].isin(sentiment_filter)]
if region_filter:
    filtered = filtered[filtered['region'].isin(region_filter)]
if period_filter:
    filtered = filtered[filtered['period'].isin(period_filter)]

# --- Display table as HTML ---
st.success(f"✅ Showing {len(filtered)} matching hashtags")

show_cols = [col for col in ['tag','sentiment','count','region','date','period','rank','time','hour'] if col in filtered.columns]
filtered = filtered[show_cols].reset_index(drop=True)
html_table = filtered.to_html(index=False, classes="custom-table", escape=False)
st.markdown(html_table, unsafe_allow_html=True)

# --- Download button ---
st.download_button(
    "📥 Download filtered data as CSV",
    data=filtered.to_csv(index=False).encode(),
    file_name="hashtags_filtered.csv",
    mime="text/csv"
)
