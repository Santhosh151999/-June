import streamlit as st
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="#June",
    layout="wide"
)

# ---------------- HIDE STREAMLIT UI ----------------
hide_st_style = """
<style>

/* Hide Streamlit menu */
#MainMenu {
    display: none;
}

/* Hide header */
header {
    display: none;
}

/* Hide footer */
footer {
    display: none;
}

/* Hide toolbar */
[data-testid="stToolbar"] {
    display: none;
}

/* Hide top decoration */
[data-testid="stDecoration"] {
    display: none;
}

/* Hide status widget */
[data-testid="stStatusWidget"] {
    display: none;
}

/* Remove top padding */
.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    max-width: 100vw;
}

/* Full background */
html, body, .stApp {
    min-height: 100vh;
    background: linear-gradient(
        120deg,
        #171930 0%,
        #303fa2 70%,
        #412e56 100%
    );
    font-family: -apple-system, BlinkMacSystemFont,
                 Segoe UI, Arial, sans-serif;
    color: #e8f1fd;
    scroll-behavior: smooth;
}

.hero {
    min-height: 92vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

/* Main title */
.june-title {
    font-size: 7rem;
    font-weight: 900;
    letter-spacing: 0.02em;
    background: linear-gradient(
        90deg,
        #65f6ff 0%,
        #668dff 60%,
        #fffadf 100%
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;

    text-align: center;
    margin-bottom: 0.6rem;

    filter: drop-shadow(0 3px 32px #baffff22);
}

/* Tagline */
.june-tagline {
    font-size: 2.2rem;
    font-weight: 450;
    text-align: center;
    color: #cdf9ff;
    margin-bottom: 3rem;
    text-shadow: 0 8px 40px #0076be24;
}

/* Button */
.cta-button {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(
        90deg,
        #71fbff,
        #868aff 90%
    );

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
    background: linear-gradient(
        90deg,
        #d1fbff,
        #dedbff 100%
    );
    color: #000;
}

/* About section */
.about-block {
    margin: 6rem auto 4rem auto;
    max-width: 760px;
    padding: 3rem 2rem;

    background: rgba(255, 255, 255, 0.05);

    border-radius: 2em;
    backdrop-filter: blur(15px);

    box-shadow: 0 8px 30px #121a2f35;

    text-align: center;

    font-family: -apple-system,
                 BlinkMacSystemFont,
                 Segoe UI,
                 Arial,
                 sans-serif;
}

</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">

    <div class="june-title">
        #June
    </div>

    <div class="june-tagline">
        Your window to world sentiment.
    </div>

    <a href="#about" class="cta-button">
        What's June?
    </a>

</div>
""", unsafe_allow_html=True)

# ---------------- FETCH TRENDING TAGS ----------------
@st.cache_data(ttl=600)
def fetch_tags_cached(url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        res = requests.get(
            url,
            headers=headers,
            timeout=5
        )

        res.raise_for_status()

        soup = BeautifulSoup(
            res.text,
            "html.parser"
        )

        tags = []

        for tr in soup.select("table.ranking tr"):

            tag_el = tr.select_one("td.main a")

            if tag_el:

                tag = tag_el.get_text(strip=True)

                if tag.startswith("#"):
                    tags.append(tag)

            if len(tags) >= 30:
                break

        return tags

    except:
        return []

# ---------------- CLEAN TEXT ----------------
def clean_english(text):
    return re.sub(
        r"[^#A-Za-z0-9_]",
        "",
        text
    )

# ---------------- GET TAGS ----------------
urls = [
    "https://getdaytrends.com",
    "https://getdaytrends.com/india"
]

all_tags = set()

for url in urls:

    tags = fetch_tags_cached(url)

    cleaned = [
        clean_english(t)
        for t in tags
        if clean_english(t)
    ]

    all_tags.update(cleaned)

# Add main brand
all_tags.add("#June")

# Frequencies
freqs = {
    tag: 1
    for tag in all_tags
}

freqs["#June"] = 100

# ---------------- FONT ----------------
FONT_PATH = "NotoSans-Regular.ttf"

if not os.path.isfile(FONT_PATH):

    st.warning(
        "Download 'NotoSans-Regular.ttf' "
        "and place it in the app folder "
        "for better rendering."
    )

# ---------------- GENERATE WORD CLOUD ----------------
@st.cache_data(ttl=600)
def generate_wordcloud(freqs, font_path):

    wc = WordCloud(
        width=900,
        height=450,
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

wc = generate_wordcloud(
    freqs,
    FONT_PATH
)

# ---------------- DISPLAY WORD CLOUD ----------------
fig, ax = plt.subplots(
    figsize=(8, 4),
    dpi=120
)

ax.imshow(
    wc,
    interpolation="bilinear"
)

ax.axis("off")

fig.patch.set_alpha(0)

plt.tight_layout(pad=0)

st.pyplot(
    fig,
    use_container_width=True
)

# ---------------- ABOUT SECTION ----------------
st.markdown("""
<div id="about" class="about-block">

    <div style="
        font-size: 2.5rem;
        font-weight: 990;

        background: linear-gradient(
            93deg,
            #59fcff 15%,
            #afdfff 60%,
            #ffe7f9 98%
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;

        margin-bottom: 1.1rem;
    ">
        #June
    </div>

    <p style="
        font-size: 1.17rem;
        font-weight: 650;
        color: #e9feff;
    ">
        #June is your live window to world conversations —
        capturing what’s buzzing across regions and languages
        in real time.
    </p>

    <p style="
        font-size: 1.17rem;
        font-weight: 650;
        color: #e9feff;
    ">
        It gathers trending hashtags from India and around
        the globe every hour and analyzes them using
        multilingual AI to understand public sentiment.
    </p>

    <p style="
        font-size: 1.17rem;
        font-weight: 650;
        color: #e9feff;
    ">
        Whether it’s joy, concern, celebration,
        or movement — #June helps you feel the pulse
        of the world beautifully and clearly.
    </p>

    <p style="
        font-size: 1.17rem;
        font-weight: 650;
        color: #e9feff;
    ">
        Made for curious minds, researchers,
        creators, and explorers —
        #June transforms social chatter
        into meaningful insight.
    </p>

</div>
""", unsafe_allow_html=True)
