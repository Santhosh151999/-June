import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import pytz
import re
import plotly.express as px
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')

# --- THEME AND BACKGROUND ---
st.set_page_config(page_title="Sentiment • #June", layout="wide")
st.markdown("""
<style>
html, body, .stApp {
    height: 100%;
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
    box-shadow: 0 8px 28px rgba(33,55,95,0.20), 0 1.5px 14px rgba(0,0,0,0.05);
    border-radius: 22px !important;
    font-size: 1.08rem !important;
    color: #ddf6ff !important;
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
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("table.ranking tr"):
        pos = tr.select_one("th.pos")
        main = tr.select_one("td.main")
        if not pos or not main:
            continue
        try:
            rank = int(pos.get_text(strip=True))
        except:
            rank = 0
        tag_el = main.select_one("a")
        tag_text = tag_el.get_text(strip=True) if tag_el else main.get_text(strip=True)
        desc_el = main.select_one("div.desc")
        desc_text = desc_el.get_text(strip=True) if desc_el else ""
        rows.append({
            "tag": tag_text.lower(),
            "rank": rank,
            "tweet_text": desc_text,
            "region": region,
            "datetime": target_dt,
            "date": target_dt.date(),
            "hour": target_dt.hour,
            "hour_str": f"{target_dt.hour:02}:00"
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
    if suffix == "k":
        return int(val * 1000)
    elif suffix == "m":
        return int(val * 1000000)
    else:
        return int(val)

# Set IST time
ist = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(ist).replace(minute=0, second=0, microsecond=0)

progress = st.progress(0)
status = st.empty()

status.text("Fetching current hour data...")
world_now = extract_trends("https://getdaytrends.com/", now_ist, "World")
india_now = extract_trends("https://getdaytrends.com/india/", now_ist, "India")
progress.progress(0.15)

status.text("Fetching past 23 hours...")
world_past, india_past = [], []
for i in range(1, 24):
    dt = now_ist - timedelta(hours=i)
    world_past += extract_trends(f"https://getdaytrends.com/{i}/", dt, "World")
    india_past += extract_trends(f"https://getdaytrends.com/india/{i}/", dt, "India")
    progress.progress(0.15 + (i/23)*0.85)

df = pd.DataFrame(world_now + india_now + world_past + india_past)

if df.empty:
    st.error("No trending data available.")
    st.stop()

df["tweet_count"] = df["tweet_text"].apply(parse_tweet_count)
df["rank"] = df["rank"].fillna(0).astype(int)
df["region"] = df["region"].str.title()
df["datetime"] = pd.to_datetime(df["datetime"])
df["date"] = df["date"].astype(str)
df["hour_str"] = df["hour_str"].astype(str)

# ──────────────────────────────────────
# 🔥 FAST multilingual sentiment model
# ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", device=-1)

model = load_sentiment_model()

# Batched inference for speed
sentiment_cache = {}
batch_size = 32
unique_tags = df["tag"].unique()

for i in range(0, len(unique_tags), batch_size):
    batch = unique_tags[i:i+batch_size]
    texts = [tag.replace("#", "").replace("_", " ") for tag in batch]
    try:
        preds = model(texts)
        for tag, pred in zip(batch, preds):
            sentiment_cache[tag] = pred["label"]
    except Exception:
        for tag in batch:
            sentiment_cache[tag] = "Neutral"

df["sentiment"] = df["tag"].map(sentiment_cache)

# ───────────── Filtering and Display ─────────────
search_text = st.text_input("Search hashtags:")
if search_text.strip():
    filtered_df = df[df["tag"].str.contains(search_text.strip().lower(), na=False)]
else:
    filtered_df = df.copy()

cutoff = now_ist - timedelta(hours=23)
recent_df = filtered_df[(filtered_df["datetime"] >= cutoff) & (filtered_df["datetime"] < now_ist)]
current_df = filtered_df[
    (filtered_df["datetime"].dt.date == now_ist.date()) &
    (filtered_df["datetime"].dt.hour == now_ist.hour)
]

current_hour_label = now_ist.strftime("%H:00")
current_date_label = now_ist.strftime("%Y-%m-%d")

st.success(f"{len(filtered_df)} hashtags loaded • Latest: {current_date_label} {current_hour_label}")

def show_top_tags(section_df, title):
    st.subheader(title)
    col1, col2 = st.columns(2)
    for region, col in zip(["World", "India"], [col1, col2]):
        region_df = section_df[section_df["region"] == region]
        if region_df.empty:
            col.info(f"No {region} data.")
            continue
        top10 = (region_df.groupby("tag").agg({
            "tweet_count": "sum",
            "sentiment": "first",
            "date": "first",
            "hour_str": "first",
            "rank": "first",
            "region": "first"
        }).reset_index().sort_values("tweet_count", ascending=False).head(10))
        col.dataframe(top10, use_container_width=True, height=420)

show_top_tags(recent_df, "Top 10 Hashtags (Last 23 Hours)")
show_top_tags(current_df, f"Top 10 Hashtags (Current Hour: {current_hour_label})")

# ───────────── Visuals ─────────────
st.divider()
st.header("Sentiment Distribution")

for region in ["World", "India"]:
    region_data = filtered_df[filtered_df["region"] == region]
    if region_data.empty:
        st.info(f"No data for {region}")
        continue
    fig = px.histogram(
        region_data, x="sentiment", color="sentiment",
        category_orders={"sentiment": ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]},
        color_discrete_map={
            "Very Positive": "green", "Positive": "lime",
            "Neutral": "gray", "Negative": "orange", "Very Negative": "red"
        },
        title=f"{region} Sentiment Distribution"
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)

st.header(f"Current Hour Sentiment ({current_hour_label} {current_date_label})")
for region in ["World", "India"]:
    region_data = current_df[current_df["region"] == region]
    if region_data.empty:
        st.info(f"No data for {region}")
        continue
    counts = region_data["sentiment"].value_counts().reindex(
        ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
    ).fillna(0)
    fig = px.pie(
        counts, names=counts.index, values=counts.values,
        color_discrete_map={
            "Very Positive": "green", "Positive": "lime",
            "Neutral": "gray", "Negative": "orange", "Very Negative": "red"
        },
        title=f"{region} Sentiment (Current Hour)", hole=0.4
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
    st.plotly_chart(fig)

# ───────────── Download ─────────────
csv = filtered_df[["tag", "tweet_count", "sentiment", "date", "hour_str", "rank", "region"]].to_csv(index=False).encode()
st.header("Download Filtered Data")
st.download_button("Download CSV", csv, "hashtags_sentiment.csv")
