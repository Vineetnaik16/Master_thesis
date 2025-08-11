# Home.py  (root, not in /pages)
from pathlib import Path
import base64
import streamlit as st

st.set_page_config(page_title="BullVision - Main Menu", page_icon="🐂", layout="wide")

ASSETS = Path("assets")
LOGO_PATH = ASSETS / "bullvision_logo.png"
BG_PATH   = ASSETS / "bull_bear_bg.png"  # use .png here (or switch to .jpg and update this path)

def bg(img: Path):
    if img.exists():
        b64 = base64.b64encode(img.read_bytes()).decode()
        ext = img.suffix[1:]
        st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,.55), rgba(0,0,0,.55)),
                        url("data:image/{ext};base64,{b64}") center/cover fixed no-repeat;
        }}
        .hero {{
            max-width: 960px;
            margin: 7vh auto 5vh;
            padding: 2.2rem;
            border-radius: 20px;
            background: rgba(255,255,255,.9);
            box-shadow: 0 20px 48px rgba(0,0,0,.25);
            text-align: center;
        }}
        .hero h1 {{
            font-size: 3.2rem;
            margin: .2rem 0;
            color: #0f172a;
        }}
        .hero p {{
            margin: 0;
            font-size: 1.2rem;
            color: #334155;
        }}
        /* Bigger primary CTA */
        .stButton>button[kind="primary"] {{
            font-size: 1.15rem;
            padding: .9rem 1.2rem;
            border-radius: 12px;
        }}
        </style>
        """, unsafe_allow_html=True)

def logo(img: Path, h: int = 72) -> str:
    if not img.exists():
        return ""
    b64 = base64.b64encode(img.read_bytes()).decode()
    ext = img.suffix[1:]
    return f'<img src="data:image/{ext};base64,{b64}" alt="BullVision" style="height:{h}px;margin-bottom:10px;">'

bg(BG_PATH)

st.markdown(f"""
<div class="hero">
  {logo(LOGO_PATH)}
  <h1>BullVision</h1>
  <p>Shaping strong investment strategies</p>
</div>
""", unsafe_allow_html=True)

def go(page: str):
    try:
        st.switch_page(page)  # page path must be relative to this file and live in /pages
    except Exception:
        st.page_link(page, label="Open →")

# --- Single centered CTA ---
left, center, right = st.columns([1, 2, 1])
with center:
    if st.button("Let’s Go 🚀", type="primary", use_container_width=True):
        go("pages/01_Market_Data_Scraper.py")

st.caption("Images expected in ./assets/bullvision_logo.png and ./assets/bull_bear_bg.png")
