import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import pytz
import re
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

# ───── STREAMLIT STYLING ─────
st.set_page_config(page_title="Hashtags • #June", layout="wide")
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(135deg, #243B55 0, #141e30 40%, #0a1531 100%) !important;
    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif;
    color: #eaf7ff;
}
.block-container { max-width: 1450px; padding: 1.5rem 3rem; margin: auto; }
h1, h2, h3, h4 { color: #eaf7ff !important; }
.stDataFrame, .stTable {
    background: rgba(63, 84, 120, 0.20) !important;
    box-shadow: 0 8px 28px rgba(33,55,95,0.20), 0 1.5px 14px rgba(0,0,0,0.05);
    border-radius: 22px !important;
    font-size: 1.08rem !important;
    color: #ddf6ff !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏷 Hashtag Explorer • #June")
st.caption("Real-time multilingual sentiment on trending tags")

# ───── UTILITIES ─────
def parse_count(text):
    if not isinstance(text, str): return 0
    text = text.replace(",", "").strip().lower()
    match = re.match(r'(\d+(\.\d+)?)([km])?', text)
    if not match: return 0
    num = float(match.group(1))
    suffix = match.group(3)
    if suffix == 'm': return int(num * 1_000_000)
    elif suffix == 'k': return int(num * 1_000)
    return int(num)

def clean_record_date(raw):
    now = datetime.now()
    raw = str(raw).strip().lower()
    if "min" in raw: return now.strftime('%Y-%m-%d')
    if "hour" in raw: return (now - timedelta(hours=int(re.search(r'\d+', raw).group()))).strftime('%Y-%m-%d')
    if "day" in raw: return (now - timedelta(days=int(re.search(r'\d+', raw).group()))).strftime('%Y-%m-%d')
    for fmt in ("%b %d, %Y", "%b %d, %y"):
        try: return datetime.strptime(raw, fmt).strftime('%Y-%m-%d')
        except: continue
    return now.strftime('%Y-%m-%d')

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", device=-1)

def run_sentiment(tags):
    model = load_sentiment_model()
    cache = {}
    batch_size = 32
    for i in range(0, len(tags), batch_size):
        batch = tags[i:i+batch_size]
        clean_input = [tag.replace("#", "").replace("_", " ") for tag in batch]
        try:
            results = model(clean_input)
            for tag, res in zip(batch, results):
                cache[tag] = res["label"]
        except:
            for tag in batch:
                cache[tag] = "Neutral"
    return cache

# ───── EXTRACT “NOW” DATA ─────
def extract_now(region: str):
    url = "https://getdaytrends.com/" if region.lower() == "world" else "https://getdaytrends.com/india/"
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
    except:
        return pd.DataFrame()
    soup = BeautifulSoup(html, "html.parser")
    data = []
    for i, tr in enumerate(soup.select("table.ranking tr")):
        main = tr.select_one("td.main")
        if not main: continue
        tag = main.select_one("a").get_text(strip=True) if main.select_one("a") else main.get_text(strip=True)
        desc = main.select_one("div.desc").get_text(strip=True) if main.select_one("div.desc") else ""
        match = re.search(r"([\d.,]+[KM]?)\s+Tweets", desc, re.IGNORECASE)
        count = parse_count(match.group(1)) if match else 0
        data.append({
            'tag': tag.lower(),
            'count': count,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'rank': i + 1,
            'region': region.title(),
            'period': 'Now'
        })
    return pd.DataFrame(data)

# ───── EXTRACT “TOP” HISTORY DATA ─────
def extract_top(url, region, period, cap=50):
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
    except:
        return pd.DataFrame()
    soup = BeautifulSoup(html, "html.parser")
    data = []
    for i, tr in enumerate(soup.select("table.ranking tr")):
        tds = tr.find_all("td")
        if len(tds) < 3: continue
        tag = tds[0].get_text(strip=True).lower()
        count = parse_count(tds[1].get_text(strip=True))
        date_text = next(tds[2].stripped_strings, "")
        data.append({
            'tag': tag,
            'count': count,
            'date': clean_record_date(date_text),
            'rank': i + 1,
            'region': region.title(),
            'period': period
        })
        if len(data) >= cap: break
    return pd.DataFrame(data)

# ───── DATA EXTRACTION STEP 1: "Now" ─────
st.info("📡 Extracting 'Now' data...")
progress = st.progress(0)
now_world = extract_now("World")
progress.progress(0.25)
now_india = extract_now("India")
progress.progress(0.5)

df_now = pd.concat([now_world, now_india], ignore_index=True)
sentiment_now = run_sentiment(df_now["tag"].tolist())
df_now["sentiment"] = df_now["tag"].map(sentiment_now)

# ───── DATA EXTRACTION STEP 2: Top Hashtags ─────
st.info("📊 Extracting yearly/monthly/week data...")
urls = [
    ("https://getdaytrends.com/top/tweeted/year/", "World", "Year"),
    ("https://getdaytrends.com/top/tweeted/month/", "World", "Month"),
    ("https://getdaytrends.com/top/tweeted/week/", "World", "Week"),
    ("https://getdaytrends.com/india/top/tweeted/year/", "India", "Year"),
    ("https://getdaytrends.com/india/top/tweeted/month/", "India", "Month"),
    ("https://getdaytrends.com/india/top/tweeted/week/", "India", "Week"),
]

df_top = pd.DataFrame()
for i, (url, reg, per) in enumerate(urls):
    df = extract_top(url, reg, per, 50)
    df_top = pd.concat([df_top, df], ignore_index=True)
    progress.progress(0.5 + (i+1)/len(urls) * 0.5)

sentiment_top = run_sentiment(df_top["tag"].tolist())
df_top["sentiment"] = df_top["tag"].map(sentiment_top)

# ───── COMBINE FINAL RESULTS ─────
df_all = pd.concat([df_now, df_top], ignore_index=True)
df_all["count"] = pd.to_numeric(df_all["count"], errors="coerce").fillna(0).astype(int)
df_all["rank"] = pd.to_numeric(df_all["rank"], errors="coerce").fillna(0).astype(int)
df_all["date"] = df_all["date"].astype(str)

# ───── FILTER UI ─────
col1, col2 = st.columns(2)
search = col1.text_input("🔍 Search hashtag:")
sentiment_filter = col2.multiselect("🧠 Sentiment", 
        ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"])

cnt_range = st.slider("📊 Tweet Count Range", int(df_all["count"].min()), int(df_all["count"].max()),
                      (int(df_all["count"].min()), int(df_all["count"].max())))

region_filter = st.multiselect("🌍 Region", sorted(df_all["region"].dropna().unique()), default=[])
period_filter = st.multiselect("⏱ Period", sorted(df_all["period"].unique()), default=[])

df_filtered = df_all.copy()
if search:
    df_filtered = df_filtered[df_filtered["tag"].str.contains(search.strip(), case=False)]
if sentiment_filter:
    df_filtered = df_filtered[df_filtered["sentiment"].isin(sentiment_filter)]
if region_filter:
    df_filtered = df_filtered[df_filtered["region"].isin(region_filter)]
if period_filter:
    df_filtered = df_filtered[df_filtered["period"].isin(period_filter)]
df_filtered = df_filtered[df_filtered["count"].between(*cnt_range)]

# ───── FINAL TABLE ─────
st.success(f"✅ Showing {len(df_filtered)} hashtags")
final_cols = ["tag", "sentiment", "count", "region", "date", "period", "rank"]
st.markdown(df_filtered[final_cols].to_html(index=False, classes="custom-table", escape=False), unsafe_allow_html=True)

# ───── DOWNLOAD ─────
st.download_button("📥 Download CSV", data=df_filtered[final_cols].to_csv(index=False).encode(),
                   file_name="hashtags_sentiment.csv")
