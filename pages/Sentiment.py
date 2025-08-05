import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import pytz
import re
import plotly.express as px
from transformers import pipeline
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# --- TRUE GLASSY BACKGROUND + FONT ---
st.set_page_config(page_title="Sentiment • #June", layout="wide")
st.markdown("""
<style>
html, body, .stApp {
    height: 100%;
    min-height: 100%;
    /* Original blue-violet glass gradient: */
    background: linear-gradient(135deg, #243B55 0, #141e30 40%, #0a1531 100%) !important;
    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif;
    color: #eaf7ff;
}
.block-container {
    max-width: 1450px;
    padding: 1.5rem 3rem;
    margin: auto;
}
h1, h2, h3, h4 {
    color: #eaf7ff !important;
}
.stDataFrame, .stTable {
    background: rgba(63, 84, 120, 0.20) !important;
    box-shadow: 0 8px 28px 0 rgba(33,55,95,0.20), 0 1.5px 14px rgba(0,0,0,0.05);
    border-radius: 22px !important;
    font-size: 1.08rem !important;
    color: #ddf6ff !important;
}
.stDataFrame table, .stTable table {
    width: 100% !important;
}
.stDataFrame thead tr, .stTable thead tr {
    background: rgba(24, 36, 59, 0.76) !important;
    color: #86e0fa !important;
    font-weight: 700 !important;
    font-size: 1.10rem !important;
}
.stDataFrame tbody tr, .stTable tbody tr {
    transition: background-color 0.25s ease;
}
.stDataFrame tbody tr:hover, .stTable tbody tr:hover {
    background-color: rgba(58, 108, 179, 0.30) !important;
}
.stDataFrame td, .stTable td, .stDataFrame th, .stTable th {
    color: #e3f5fc !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Sentiment Dashboard • #June")
st.caption("Live sentiment analysis of trending hashtags in World & India (IST)")

def fetch_html(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.text
    except:
        return None

def extract_trends(url, target_dt, region):
    html = fetch_html(url)
    if not html:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for tr in soup.select('table.ranking tr'):
        pos = tr.select_one('th.pos')
        main = tr.select_one('td.main')
        if not pos or not main:
            continue
        rank_text = pos.get_text(strip=True)
        try:
            rank = int(rank_text)
        except:
            rank = 0
        tag_el = main.select_one('a')
        tag_text = tag_el.get_text(strip=True) if tag_el else main.get_text(strip=True)
        desc_el = main.select_one('div.desc')
        desc_text = desc_el.get_text(strip=True) if desc_el else ""
        rows.append({
            'tag': tag_text.lower(),
            'rank': rank,
            'tweet_text': desc_text,
            'region': region,
            'datetime': target_dt,
            'date': target_dt.date(),
            'hour': target_dt.hour,
            'hour_str': f"{target_dt.hour:02}:00"
        })
        if len(rows) >= 50:
            break
    return rows

def parse_tweet_count(text):
    if not isinstance(text, str):
        return 0
    text = text.replace(",", "").strip().lower()
    m = re.search(r'(\d+(\.\d+)?)(k|m)?', text)
    if not m:
        return 0
    val = float(m.group(1))
    suffix = m.group(3)
    if suffix == 'k':
        return int(val * 1000)
    elif suffix == 'm':
        return int(val * 1000000)
    else:
        return int(val)

# Set IST timezone for everything
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist).replace(minute=0, second=0, microsecond=0)

progress = st.progress(0)
status_text = st.empty()

status_text.text("Fetching current hour data...")
world_now = extract_trends("https://getdaytrends.com/", now_ist, "World")
india_now = extract_trends("https://getdaytrends.com/india/", now_ist, "India")
progress.progress(0.15)

status_text.text("Fetching past 23 hours data for World...")
world_past = []
for i in range(1, 24):
    dt = now_ist - timedelta(hours=i)
    url = f"https://getdaytrends.com/{i}/"
    world_past.extend(extract_trends(url, dt, "World"))
    progress.progress(0.15 + (i / 24) * 0.40)

status_text.text("Fetching past 23 hours data for India...")
india_past = []
for i in range(1, 24):
    dt = now_ist - timedelta(hours=i)
    url = f"https://getdaytrends.com/india/{i}/"
    india_past.extend(extract_trends(url, dt, "India"))
    progress.progress(0.55 + (i / 24) * 0.45)

progress.progress(1.0)
status_text.text("All data fetched successfully.")

all_rows = world_now + india_now + world_past + india_past
df = pd.DataFrame(all_rows)

if df.empty:
    st.error("No data could be fetched from the sources. Please check your internet connection or site availability.")
    st.stop()

df['tweet_count'] = df['tweet_text'].apply(parse_tweet_count)
df['rank'] = df['rank'].fillna(0).astype(int)
df['region'] = df['region'].str.title()
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['date'].astype(str)
df['hour_str'] = df['hour_str'].astype(str)

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base", device=-1)

model = load_sentiment_model()
cache_file = "sentiment_cache.pkl"

if os.path.exists(cache_file):
    sentiment_cache = joblib.load(cache_file)
else:
    sentiment_cache = {}

unique_tags = df['tag'].unique()
to_predict = [tag for tag in unique_tags if tag not in sentiment_cache]

if to_predict:
    preds = model(to_predict, truncation=True)
    for tag, pred in zip(to_predict, preds):
        label = pred['label'].lower()
        if 'positive' in label:
            sentiment_cache[tag] = 'Positive'
        elif 'neutral' in label:
            sentiment_cache[tag] = 'Neutral'
        else:
            sentiment_cache[tag] = 'Negative'
    joblib.dump(sentiment_cache, cache_file)

df['sentiment'] = df['tag'].map(sentiment_cache)

search_text = st.text_input("Search hashtags:", "")
if search_text.strip():
    filtered_df = df[df['tag'].str.contains(search_text.strip().lower(), na=False)]
else:
    filtered_df = df.copy()

cutoff_time = now_ist - timedelta(hours=23)
recent_df = filtered_df[(filtered_df['datetime'] >= cutoff_time) & (filtered_df['datetime'] < now_ist)]
current_df = filtered_df[(filtered_df['datetime'].dt.hour == now_ist.hour) & (filtered_df['datetime'].dt.date == now_ist.date())]

current_hour_label = now_ist.strftime("%H:00")
current_date_label = now_ist.strftime("%Y-%m-%d")

st.success(f"Showing {len(filtered_df)} hashtags. Latest data time: {current_date_label} {current_hour_label}")

st.markdown("### Top 10 Hashtags (Last 23 Hours)")
col1, col2 = st.columns(2)
for region_name, col in zip(['World', 'India'], [col1, col2]):
    region_data = recent_df[recent_df['region'] == region_name]
    if region_data.empty:
        col.warning(f"No data for {region_name} in last 23 hours.")
        continue
    top10 = (region_data.groupby('tag')
             .agg({'tweet_count':'sum','sentiment':'first','date':'first','hour_str':'first','rank':'first','region':'first'})
             .reset_index()
             .sort_values('tweet_count', ascending=False)
             .head(10))
    col.dataframe(top10[['tag','tweet_count','sentiment','date','hour_str','rank','region']], use_container_width=True, height=420)

st.markdown(f"### Top 10 Hashtags (Current Hour: {current_hour_label} {current_date_label})")
col3, col4 = st.columns(2)
for region_name, col in zip(['World', 'India'], [col3, col4]):
    region_data = current_df[current_df['region'] == region_name]
    if region_data.empty:
        col.warning(f"No data for {region_name} in current hour.")
        continue
    top10 = (region_data.groupby('tag')
             .agg({'tweet_count':'sum','sentiment':'first','date':'first','hour_str':'first','rank':'first','region':'first'})
             .reset_index()
             .sort_values('tweet_count', ascending=False)
             .head(10))
    col.dataframe(top10[['tag','tweet_count','sentiment','date','hour_str','rank','region']], use_container_width=True, height=420)

st.divider()

st.header("Sentiment Distribution")
for region_name in ['World', 'India']:
    region_data = filtered_df[filtered_df['region'] == region_name]
    if region_data.empty:
        st.warning(f"No sentiment data for {region_name}.")
        continue
    fig = px.histogram(region_data, x='sentiment', color='sentiment',
                       category_orders={'sentiment': ['Positive', 'Neutral', 'Negative']},
                       color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
                       title=f"{region_name} Sentiment Distribution")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)

st.header(f"Current Hour Sentiment ({current_hour_label} {current_date_label})")
for region_name in ['World', 'India']:
    region_data = current_df[current_df['region'] == region_name]
    if region_data.empty:
        st.warning(f"No current hour sentiment data for {region_name}.")
        continue
    counts = region_data['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
    fig = px.pie(counts, names=counts.index, values=counts.values,
                 color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
                 title=f"{region_name} Current Hour Sentiment")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)

csv_bytes = filtered_df[['tag', 'tweet_count', 'sentiment', 'date', 'hour_str', 'rank', 'region']].to_csv(index=False).encode()
st.header("Download filtered data")
st.download_button("Download CSV", csv_bytes, "hashtags_sentiment.csv")
