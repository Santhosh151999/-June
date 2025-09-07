import streamlit as st
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os

# Streamlit page config maintains your theme
st.set_page_config(page_title="#June", layout="wide")

# Existing CSS styles - unchanged
st.markdown("""
<style>
html, body, .stApp {
    min-height: 100vh;
    background: linear-gradient(120deg, #171930 0%, #303fa2 70%, #412e56 100%);
    font-family: -apple-system,BlinkMacSystemFont, Segoe UI, Arial, sans-serif;
    color: #e8f1fd;
    scroll-behavior: smooth;
}
.block-container {
    max-width: 100vw;
    padding: 0 1.5rem;
    margin: 0 auto;
}
.hero {
    min-height: 92vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.june-title {
    font-size: 7rem;
    font-weight: 900;
    letter-spacing: 0.02em;
    background: linear-gradient(90deg,#65f6ff 0%,#668dff 60%,#fffadf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 3px 32px #baffff22);
}
.june-tagline {
    font-size: 2.2rem;
    font-weight: 450;
    text-align: center;
    color: #cdf9ff;
    margin-bottom: 3rem;
    text-shadow: 0 8px 40px #0076be24;
}
.cta-button {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #71fbff, #868aff 90%);
    color: #08131a;
    padding: 1rem 3rem;
    border: none;
    border-radius: 3rem;
    cursor: pointer;
    text-decoration: none;
    box-shadow: 0 5px 24px #61fff371;
    transition: all 0.2s ease-in-out;
    margin-bottom: 2.3rem;
}
.cta-button:hover {
    background: linear-gradient(90deg, #d1fbff, #dedbff 100%);
    color: #000;
}
.about-block {
    margin: 6rem auto 4rem auto;
    max-width: 760px;
    padding: 3rem 2rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 2em;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 30px #121a2f35;
    text-align: center;
    font-family: -apple-system,BlinkMacSystemFont, Segoe UI, Arial, sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Hero section unchanged
st.markdown("""
<div class="hero">
    <div class="june-title">#June</div>
    <div class="june-tagline">Your window to world sentiment.</div>
    <a href="#about" class="cta-button">What's June?</a>
</div>
""", unsafe_allow_html=True)

# --- Word Cloud Integration with Optimization ---

@st.cache_data(ttl=600)
def fetch_tags_cached(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        tags = []
        for tr in soup.select("table.ranking tr"):
            tag_el = tr.select_one("td.main a")
            if tag_el:
                tag = tag_el.get_text(strip=True)
                if tag.startswith("#"):
                    tags.append(tag)
            if len(tags) >= 30:  # limit tags for speed
                break
        return tags
    except:
        return []

def clean_english(text):
    return re.sub(r"[^#A-Za-z0-9_]", "", text)

urls = ["https://getdaytrends.com", "https://getdaytrends.com/india"]
all_tags = set()

for url in urls:
    tags = fetch_tags_cached(url)
    cleaned = [clean_english(t) for t in tags if clean_english(t)]
    all_tags.update(cleaned)

all_tags.add("#June")

freqs = {tag: 1 for tag in all_tags}
freqs["#June"] = 100  # biggest prominence

FONT_PATH = "NotoSans-Regular.ttf"
if not os.path.isfile(FONT_PATH):
    st.warning("Download 'NotoSans-Regular.ttf' font and place it in the app directory for best wordcloud rendering.")

@st.cache_data(ttl=600)
def generate_wordcloud(freqs, font_path):
    wc = WordCloud(
        width=800,
        height=400,
        background_color=None,
        mode="RGBA",
        colormap="tab10",
        prefer_horizontal=0.9,
        max_words=70,
        min_font_size=14,
        max_font_size=80,
        font_path=font_path if os.path.isfile(font_path) else None,
        random_state=42
    ).generate_from_frequencies(freqs)
    return wc

wc = generate_wordcloud(freqs, FONT_PATH)

fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
fig.patch.set_alpha(0)
plt.tight_layout(pad=0)

st.pyplot(fig, use_container_width=True)

# --- About section unchanged below ---
st.markdown("""
<div id="about" class="about-block" style="text-align:center;">
    <div style="
        font-size: 2.5rem;
        font-weight: 990;
        background: linear-gradient(93deg,#59fcff 15%,#afdfff 60%,#ffe7f9 98%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.1rem;
    ">
        #June
    </div>
    <p style="font-size: 1.17rem; font-weight: 650; color: #e9feff;">
        #June is your live window to the world conversations—capturing what’s buzzing amongst people across regions and languages right now. It gathers trending hashtags from India and all across the globe every hour, and analyzes them using powerful multilingual AI to understand how people feel—positive, neutral, or negative.
    </p>
    <p style="font-size: 1.17rem; font-weight: 650; color: #e9feff;">
        But #June isn't just about data or numbers—it's about capturing real emotion, real time.
    </p>
    <p style="font-size: 1.17rem; font-weight: 650; color: #e9feff;">
        With clear visuals, it helps you to truly feel the pulse of what's happening in the world around you. Whether it's a joyful trend, an important movement, or a global moment, #June makes the story clear, beautiful, and easy to explore.
    </p>
    <p style="font-size: 1.17rem; font-weight: 650; color: #e9feff;">
        Made for everyone from curious minds and researchers, #June turns complex social chatter maze into simple, uplifting insight—so you can connect with what truly matters, every moment.
    </p>
</div>
""", unsafe_allow_html=True)
