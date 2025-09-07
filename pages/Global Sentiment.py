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

# --- THEME AND BACKGROUND (from sentiment.py) ---
st.set_page_config(page_title="Global Sentiment • #June", layout="wide")
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

st.title("📊 Global Sentiment • #June")
st.caption("Filter by country and 'hours ago' to analyze trending hashtags.")

# --- Existing globalsentiment.py code ---
COUNTRY_LIST = [
    "algeria","argentina","australia","austria","bahrain","belarus","belgium",
    "brazil","canada","chile","colombia","denmark","dominican-republic","ecuador",
    "egypt","france","germany","ghana","greece","guatemala","india","indonesia",
    "ireland","israel","italy","japan","kenya","korea","kuwait","latvia","lebanon",
    "malaysia","mexico","netherlands","new-zealand","nigeria","norway","oman",
    "pakistan","panama","peru","philippines","poland","portugal","puerto-rico",
    "qatar","russia","saudi-arabia","singapore","south-africa","spain","sweden",
    "switzerland","thailand","turkey","ukraine","united-arab-emirates","united-kingdom",
    "united-states","venezuela","vietnam"
]

ist = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(ist).replace(minute=0, second=0, microsecond=0)

def fetch_html(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.text
    except:
        return None

def extract_trends(country, hour_ago):
    target_dt = now_ist - timedelta(hours=hour_ago)
    suffix = "" if hour_ago == 0 else f"{hour_ago}/"
    if country == "india":
        url = f"https://getdaytrends.com/india/{suffix}"
    else:
        url = f"https://getdaytrends.com/{country}/{suffix}"
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
            "region": country.title(),
            "datetime": target_dt,
            "date": target_dt.date(),
            "hour": target_dt.hour,
        })
        if len(rows) >= 50:
            break
    return rows

def parse_tweet_count(text):
    if not isinstance(text, str):
        return 0
    text = text.lower().strip()
    if "under 10k" in text:
        return 10000
    m = re.search(r'([\d\.]+)([km]?)', text)
    if not m:
        return 0
    val = float(m.group(1))
    suffix = m.group(2)
    if suffix == 'k':
        return int(val * 1000)
    elif suffix == 'm':
        return int(val * 1000000)
    return int(val)

@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=-1)

def map_sentiment(label):
    if label in ["1 star", "2 stars"]:
        return "Negative"
    elif label == "3 stars":
        return "Neutral"
    elif label in ["4 stars", "5 stars"]:
        return "Positive"
    label = label.lower()
    if "neg" in label:
        return "Negative"
    if "pos" in label:
        return "Positive"
    return "Neutral"

def run_sentiment(model, tags):
    cache = {}
    batch_size = 32
    for i in range(0, len(tags), batch_size):
        batch = tags[i:i+batch_size]
        texts = [tag.replace("#", " ").replace("_", " ") for tag in batch]
        try:
            results = model(texts)
            for t, res in zip(batch, results):
                cache[t] = map_sentiment(res["label"])
        except:
            for t in batch:
                cache[t] = "Neutral"
    return cache

# --- Filters ---
selected_country = st.selectbox("Select Country", options=[""] + sorted(COUNTRY_LIST))
hour_options = [""] + [f"{i} hours ago" for i in range(24)]
selected_hour_str = st.selectbox("Select Hour", options=hour_options)

if selected_country == "" or selected_hour_str == "":
    st.info("Please select both country and hour.")
    st.stop()

hour_val = int(selected_hour_str.split()[0])
progress = st.progress(0)
status = st.empty()
status.text(f"Fetching {selected_country.title()} trends {selected_hour_str}...")

rows = extract_trends(selected_country.lower(), hour_val)
progress.progress(1.0)
if not rows:
    st.warning("No trends found.")
    st.stop()

df = pd.DataFrame(rows)
df["tweet_count"] = df["tweet_text"].apply(parse_tweet_count)
df.drop(columns=["tweet_text"], inplace=True)
df["rank"] = df["rank"].fillna(0).astype(int)
df["region"] = df["region"].astype(str).str.title()
df["date"] = df["date"].astype(str)

model = load_model()
sent_map = run_sentiment(model, df["tag"].unique())
df["sentiment"] = df["tag"].map(sent_map)

st.subheader(f"Hashtags in {selected_country.title()} for {selected_hour_str}")
st.dataframe(df[["tag", "tweet_count", "sentiment", "rank"]].sort_values("tweet_count", ascending=False).reset_index(drop=True))

# --- Line plot: Hashtags vs Tweet Count ---
fig_line = px.line(df.sort_values("tweet_count", ascending=False),
                   x="tag", y="tweet_count",
                   markers=True,
                   labels={"tag":"Hashtag","tweet_count":"Tweet Count"},
                   title="Tweet Counts per Hashtag")
st.plotly_chart(fig_line, use_container_width=True)

# --- Bar chart: Sentiment vs Count ---
sent_counts = df["sentiment"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0).reset_index()
sent_counts.columns = ["Sentiment","Count"]
fig_bar = px.bar(sent_counts, x="Sentiment", y="Count",
                 color="Sentiment",
                 color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"},
                 title="Sentiment Distribution")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Download CSV ---
csv_data = df.to_csv(index=False)
st.download_button("Download CSV", csv_data.encode(), f"{selected_country}_hour{hour_val}_trends.csv")
