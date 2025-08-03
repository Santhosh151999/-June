import streamlit as st
import os
import pandas as pd
import plotly.express as px
from transformers import pipeline
from datetime import datetime, timedelta

# --- THEME: Unified June Style, Glassy Tables ---
st.set_page_config(page_title="Sentiment • #June", layout="wide")
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(120deg, #171930 0%, #303fa2 70%, #412e56 100%);
    font-family: -apple-system,BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif;
    color: #e8f1fd;
}
.block-container {
    padding: 1rem 2rem;
    max-width: 100vw;
    margin: 0 auto;
}
.stTitle, .stHeader, .stSubheader, .stCaption, .stText, .stMarkdown, .stMetric {
    color: #e8f1fd !important;
    font-family: -apple-system,BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif !important;
}
.stDataFrame, .stTable, .stElementTable, .stDataFrame div, .stTable div {
    border-radius: 18px !important;
    background: rgba(42, 49, 69, 0.5) !important;
    box-shadow: 0px 7px 34px #101a2f36, 0px 1.5px 27px #4b5a6e25;
    font-variant-numeric: tabular-nums;
}
.stDataFrame thead tr, .stTable thead tr {
    background: rgba(35, 38, 58, 0.5) !important;
    color: #87e8fb !important;
    font-weight: 720;
    font-size: 1.08rem;
}
.stDataFrame tbody tr, .stTable tbody tr {
    background: rgba(36, 40, 58, 0.25) !important;
    border-bottom: 1.2px solid #3e5475;
}
.stDataFrame tbody td, .stTable tbody td, .stDataFrame th, .stTable th {
    color: #ebf7ff !important;
    font-weight: 500 !important;
}
.stDataFrame tbody tr:hover, .stTable tbody tr:hover {
    background: rgba(50, 60, 82, 0.5) !important;
}
iframe[title="dataframe"] { background: transparent !important;}
[data-testid="stDataFrame"] { background: transparent !important;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Multilingual Sentiment Dashboard — #June")
st.caption("Live hashtag emotion detection across regions 🌏 | 🇮🇳")

# --- LOAD DATA ---
folder = "Trend_Now"
if not os.path.exists(folder):
    st.error(f"❌ Folder '{folder}' not found!")
    st.stop()

csvs = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
if not csvs:
    st.warning(f"⚠️ No CSV files found in '{folder}'. Please add CSVs.")
    st.stop()

dfs = []
for f in csvs:
    df = pd.read_csv(os.path.join(folder, f))
    df = df.rename(columns={
        "Hashtag": "tag",
        "Cleaned Tweet Count": "count",
        "Tweet Count": "count",
        "Scraped Date": "date",
        "Scraped Time": "time",
        "Hour Offset": "hour",
        "Rank": "rank",
        "Source": "region"
    })
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df['tag'] = df['tag'].astype(str).str.strip().str.lower()
df['region'] = df['region'].astype(str).str.strip().str.lower().map({'india':'India', 'world':'World'}).fillna("Unknown")
df = df[df['region'].isin(['India', 'World'])]
df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype(int)
df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)

# --- Datetime unification ---
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
df.dropna(subset=['datetime'], inplace=True)

# --- Recent data window ---
latest_ts = df['datetime'].max()
start_24h = latest_ts - timedelta(hours=24)
current_hr = latest_ts.hour
recent_date = latest_ts.date()

st.success(f"✅ Loaded {df.shape[0]} rows from {len(csvs)} file(s) — Latest: {latest_ts:%Y-%m-%d %H:%M}")

# --- SENTIMENT ANALYSIS ---
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
st.info("🔍 Running sentiment classification on hashtags...")
sent_pipe = load_sentiment_model()
unique_tags = df['tag'].unique().tolist()
sent_map = {}
results = sent_pipe(unique_tags, truncation=True)
for tag, res in zip(unique_tags, results):
    stars = ''.join(filter(str.isdigit, res['label']))
    if stars in ['4', '5']:
        sent_map[tag] = "Positive"
    elif stars == '3':
        sent_map[tag] = "Neutral"
    else:
        sent_map[tag] = "Negative"
df['sentiment'] = df['tag'].map(sent_map)

# --- SEARCH ---
st.header("🔎 Search Hashtags")
q = st.text_input("Search (partial match, multilingual supported):")
if q:
    matches = df[df['tag'].str.contains(q.strip(), case=False, na=False)]
    st.success(f"✅ Found {len(matches)} matching hashtags.")
    st.dataframe(matches[['tag', 'count', 'sentiment', 'date', 'time', 'hour', 'rank', 'region']], use_container_width=True)

st.divider()

# --- TOP HASHTAGS TABLES ---
st.header("🏆 Top 10 Trending Hashtags")
# --- Last 24 Hours ---
st.subheader("Last 24 Hours")
col1, col2 = st.columns(2)
for region, col in zip(["India", "World"], [col1, col2]):
    data = df[(df['region'] == region) & (df['datetime'] >= start_24h)]
    if data.empty:
        col.warning("No data for this region.")
        continue
    top_24h = (
        data.groupby('tag', as_index=False)['count']
        .sum()
        .sort_values('count', ascending=False)
        .head(10)
    )
    latest_meta = data.sort_values(['tag', 'datetime'], ascending=[True, False]).drop_duplicates('tag')[
        ['tag', 'sentiment', 'date', 'time', 'hour', 'rank', 'region']]
    top_24h = pd.merge(top_24h, latest_meta, on='tag', how='left')
    col.dataframe(top_24h[['tag', 'count', 'sentiment', 'date', 'time', 'hour', 'rank', 'region']], use_container_width=True)

# --- Current Hour ---
st.subheader(f"Current Hour (Hour {current_hr}, Date {recent_date})")
col1, col2 = st.columns(2)
for region, col in zip(["India", "World"], [col1, col2]):
    data = df[(df['region'] == region) & (df['datetime'].dt.hour == current_hr) & (df['datetime'].dt.date == recent_date)]
    if data.empty:
        col.warning("No data for current hour in this region.")
        continue
    top = (
        data.groupby('tag', as_index=False)['count']
        .sum()
        .sort_values('count', ascending=False)
        .head(10)
    )
    latest_meta = data.sort_values(['tag', 'datetime'], ascending=[True, False]).drop_duplicates('tag')[
        ['tag', 'sentiment', 'date', 'time', 'hour', 'rank', 'region']]
    top = pd.merge(top, latest_meta, on='tag', how='left')
    col.dataframe(top[['tag', 'count', 'sentiment', 'date', 'time', 'hour', 'rank', 'region']], use_container_width=True)

st.divider()

# --- CHARTS ---

# Sentiment Distribution
st.header("📊 Sentiment Distribution by Region")
for region in ["India", "World"]:
    st.subheader(region)
    region_data = df[df['region'] == region]
    if region_data.empty:
        st.warning("No data for this region.")
        continue
    fig = px.histogram(
        region_data,
        x="sentiment",
        color="sentiment",
        opacity=0.75,
        color_discrete_map={"Positive": "#4ed985", "Neutral": "#a8c4f4", "Negative": "#fc7676"}
    )
    fig.update_layout(title=f"{region} Sentiment Breakdown",
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# Hourly Sentiment Trend
st.header("📈 Hourly Sentiment Trend")
for region in ["India", "World"]:
    st.subheader(region)
    trend = df[df['region'] == region].groupby([df['datetime'].dt.hour, "sentiment"]).size().reset_index(name="count")
    if trend.empty:
        st.warning("No data for hourly trend in this region.")
        continue
    fig = px.line(
        trend, x="datetime", y="count", color="sentiment",
        markers=True,
        color_discrete_map={"Positive": "#4ed985", "Neutral": "#a8c4f4", "Negative": "#fc7676"},
        title=f"{region} Sentiment by Hour"
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# Pie Chart Current Hour
st.header(f"🥧 Current Hour Sentiment — Hour {current_hr}, Date {recent_date}")
col1, col2 = st.columns(2)
for region, col in zip(["India", "World"], [col1, col2]):
    data = df[(df['region'] == region) & (df['datetime'].dt.hour == current_hr) & (df['datetime'].dt.date == recent_date)]
    if data.empty:
        col.warning("No data for current hour in this region.")
        continue
    counts = data['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
    fig_pie = px.pie(
        names=counts.index,
        values=counts.values,
        color=counts.index,
        hole=0.4,
        title=f"{region} Sentiment Now",
        color_discrete_map={"Positive": "#4ed985", "Neutral": "#a8c4f4", "Negative": "#fc7676"}
    )
    fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    col.plotly_chart(fig_pie, use_container_width=True)

# --- DOWNLOAD BUTTON ---
st.header("📥 Download Full Data")
st.download_button(
    "⬇️ Download CSV",
    data=df.to_csv(index=False).encode(),
    file_name="sentiment_multilingual_data.csv",
    mime="text/csv"
)




