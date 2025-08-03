import streamlit as st

st.set_page_config(page_title="#June", layout="wide")

# --- Styles ---
st.markdown("""
<style>
html, body, .stApp {
    min-height: 100vh;
    background: linear-gradient(120deg, #171930 0%, #303fa2 70%, #412e56 100%);
    font-family: -apple-system,BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif;
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
    color: #cdf9ff;
    font-weight: 450;
    text-align: center;
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
    font-family: -apple-system,BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.info("👈 Use the sidebar to explore Sentiment, Hashtags, and more.")

# --- Hero Section with anchor scroll link ---
st.markdown("""
<div class="hero">
    <div class="june-title">#June</div>
    <div class="june-tagline">Your window to world sentiment.</div>
    <a href="#about" class="cta-button">What's June?</a>
</div>
""", unsafe_allow_html=True)

# --- About Section (user's updated content) ---
st.markdown("""
<div id="about" class="about-block" style="text-align:center;">
    <div style="
        font-size:2.5rem;
        font-weight:990;
        background:linear-gradient(93deg,#59fcff 15%,#afdfff 60%,#ffe7f9 98%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip:text;
        margin-bottom: 1.1rem;
    ">
        #June
    </div>
    <p style="font-size:1.17rem; font-weight:650; color:#e9feff;">
        #June is your live window to the world conversations—capturing what’s buzzing amongst people across regions and languages right now. It gathers trending hashtags from India and all across the globe every hour, and analyzes them using powerful multilingual AI to understand how people feel—positive, neutral, or negative.
    </p>
    <p style="font-size:1.17rem; font-weight:650; color:#e9feff;">
        But #June isn't just about data or numbers—it's about capturing real emotion, real time.
    </p>
    <p style="font-size:1.17rem; font-weight:650; color:#e9feff;">
        With clear visuals, it helps you to truly feel the pulse of what's happening in the world around you. Whether it's a joyful trend, an important movement, or a global moment, #June makes the story clear, beautiful, and easy to explore.
    </p>
    <p style="font-size:1.17rem; font-weight:650; color:#e9feff;">
        Made for everyone from curious minds and researchers, #June turns complex social chatter maze into simple, uplifting insight—so you can connect with what truly matters, every moment.
    </p>
</div>
""", unsafe_allow_html=True)
