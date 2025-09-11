import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import pytz
from datetime import datetime
from transformers import pipeline
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

warnings.filterwarnings("ignore")

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

# --- Database Setup ---
engine_no_db = create_engine("postgresql+psycopg2://Santhosh151999:Briyani2025@host:port/June")
with engine_no_db.connect() as conn:
    conn.execute(text("CREATE DATABASE IF NOT EXISTS June CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    conn.commit()

engine = create_engine("postgresql+psycopg2://Santhosh151999:Briyani2025@host:port/June")

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS Subscription (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            phone VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

# --- Helpers ---
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def add_resignation(name, email, phone):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO Subscription (name, email, phone) VALUES (:name, :email, :phone)"),
                {"name": name, "email": email, "phone": phone}
            )
        return "success"
    except Exception as e:
        if "Duplicate entry" in str(e):
            return "exists"
        else:
            st.error(f"Database error: {str(e)}")
            return "error"

def delete_resignation(name, email):
    try:
        with engine.begin() as conn:
            res = conn.execute(
                text("DELETE FROM Subscription WHERE name = :name AND email = :email"),
                {"name": name, "email": email}
            )
        if res.rowcount > 0:
            return "deleted"
        else:
            return "not_found"
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return "error"

def fetch_registered_users():
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT name, email, phone, created_at FROM Subscription ORDER BY created_at DESC"), conn)
    return df

# --- Email Sending Function (Gmail App Password) ---
def send_email_smtp(subject, body, recipients):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "ssubramaniyam0@gmail.com"   # <-- replace with your Gmail
    sender_password = "unpr ibcz tllr pmea"           # <-- your App Password without spaces

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        
        for recipient in recipients:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))  # HTML format
            server.send_message(msg)

        server.quit()
        return len(recipients)
    except Exception as e:
        st.error(f"Failed to send emails: {e}")
        return 0

# --- Trending Hashtags Helpers ---
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

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", device=-1)

def run_sentiment(model, tags):
    cache = {}
    batch_size = 32
    for i in range(0, len(tags), batch_size):
        batch = tags[i:i+batch_size]
        texts = [tag.replace("#", "").replace("_", " ") for tag in batch]
        try:
            results = model(texts)
            for t, res in zip(batch, results):
                cache[t] = res["label"]
        except:
            for t in batch:
                cache[t] = "Neutral"
    return cache

# --- Session state flags ---
for key in ["delete_mode","login_mode","admin_logged_in","show_email_template","email_content","emails_sent"]:
    if key not in st.session_state:
        st.session_state[key] = False if "mode" in key or "emails_sent" in key else ""

# --- Page config ---
st.set_page_config(page_title="Subscription", layout="wide")

# --- Admin Login button ---
cols = st.columns([9,1])
with cols[1]:
    if not st.session_state["login_mode"] and not st.session_state["admin_logged_in"]:
        if st.button("Admin Login"):
            st.session_state["login_mode"] = True

# --- Login form ---
if st.session_state["login_mode"] and not st.session_state["admin_logged_in"]:
    st.header("🔐 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username=="admin" and password=="admin":
            st.session_state["admin_logged_in"]=True
            st.session_state["login_mode"]=False
            st.success("Login successful ✅")
        else:
            st.error("Invalid credentials")
    st.stop()

# --- Admin Dashboard ---
if st.session_state["admin_logged_in"]:
    st.title("📊 Sentiment Dashboard • #June")

    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist).replace(minute=0, second=0, microsecond=0)

    st.info("Fetching current data ...")
    world_now = extract_trends("https://getdaytrends.com/", now_ist, "World")
    india_now = extract_trends("https://getdaytrends.com/india/", now_ist, "India")

    df = pd.DataFrame(world_now+india_now)
    if df.empty:
        st.error("No data available")
        st.stop()

    df["tweet_count"] = df["tweet_text"].apply(parse_tweet_count)
    df["rank"] = df["rank"].fillna(0).astype(int)
    df["region"] = df["region"].str.title()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["date"].astype(str)
    df["hour_str"] = df["hour_str"].astype(str)

    # Sentiment analysis
    model = load_sentiment_model()
    sent_cache = run_sentiment(model, df["tag"].unique())
    df["sentiment"] = df["tag"].map(sent_cache)

    search_text = st.text_input("Search hashtags:")
    filtered_df = df[df["tag"].str.contains(search_text.strip(), case=False)] if search_text.strip() else df.copy()

    st.dataframe(filtered_df[["tag","tweet_count","sentiment","region","date","rank"]].sort_values("tweet_count",ascending=False))
    st.download_button("Download CSV", filtered_df[["tag","tweet_count","sentiment","region","date","rank"]].to_csv(index=False).encode(), "hashtags.csv")

    # --- Buttons ---
    st.subheader("📋 Subscription / Email Actions")
    col1,col2=st.columns(2)
    with col1:
        if st.button("Show Subscribed Users"):
            df_users=fetch_registered_users()
            if df_users.empty: st.warning("No Subscribed users found")
            else: st.dataframe(df_users)

    with col2:
        if st.button("Fetch Data & Send Email"):
            top_india_tags = df[df["region"]=="India"].sort_values("rank").head(10)["tag"].tolist()
            top_world_tags = df[df["region"]=="World"].sort_values("rank").head(10)["tag"].tolist()
            email_html = f"""
            <h2>📊 Top Trending Hashtags</h2>
            <h3>India</h3>
            <table border="1" cellpadding="5">
                <tr><th>Rank</th><th>Hashtag</th></tr>
                {''.join([f'<tr><td>{i+1}</td><td>{tag}</td></tr>' for i, tag in enumerate(top_india_tags)])}
            </table>
            <h3>World</h3>
            <table border="1" cellpadding="5">
                <tr><th>Rank</th><th>Hashtag</th></tr>
                {''.join([f'<tr><td>{i+1}</td><td>{tag}</td></tr>' for i, tag in enumerate(top_world_tags)])}
            </table>
            """
            st.session_state["email_content"]=email_html
            st.session_state["show_email_template"]=True
            st.session_state["emails_sent"]=False

    # Show template + send email
    if st.session_state["show_email_template"]:
        st.subheader("📧 Email Template Preview")
        st.markdown(st.session_state["email_content"], unsafe_allow_html=True)
        if st.button("Send Email"):
            df_users = fetch_registered_users()
            emails = df_users["email"].tolist()
            count = send_email_smtp("Top Trending Hashtags", st.session_state["email_content"], emails)
            if count>0:
                st.success(f"✅ Emails successfully sent to {count} users")
            else:
                st.error("Failed to send emails")
            st.session_state["show_email_template"]=False

    if st.button("Logout"):
        st.session_state["admin_logged_in"]=False

    st.stop()

# --- Main User Page ---
st.title("✍️ Subscription Form")
with st.form("Subscription_form"):
    name=st.text_input("Name")
    email=st.text_input("Email")
    phone=st.text_input("Phone")
    submit=st.form_submit_button("Submit")
    if submit:
        if not name or not email or not phone: st.error("Please fill all fields")
        elif not is_valid_email(email): st.error("Invalid email")
        else:
            result=add_resignation(name,email,phone)
            if result=="success": st.success("Subscribed successfully ✅")
            elif result=="exists": st.warning("Email already registered")
            else: st.error("Database error")

# Delete record toggle
if not st.session_state["delete_mode"]:
    if st.button("Delete Record"):
        st.session_state["delete_mode"]=True

if st.session_state["delete_mode"]:
    st.header("🗑️ UnSubscription Record")
    with st.form("delete_form"):
        del_name=st.text_input("Enter Name")
        del_email=st.text_input("Enter Email")
        del_submit=st.form_submit_button("Delete")
        if del_submit:
            if not del_name or not del_email: st.error("Please enter name and email")
            else:
                result=delete_resignation(del_name,del_email)
                if result=="deleted":
                    st.success("UnSubscribed successfully ✅")
                    st.session_state["delete_mode"]=False
                elif result=="not_found":
                    st.warning("No matching record found")
