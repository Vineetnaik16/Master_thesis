# pages/00_Main_Menu.py
from pathlib import Path
import base64
import streamlit as st

st.set_page_config(page_title="BullVision • Main Menu", page_icon="🐂", layout="wide")

# ---------- Assets ----------
ASSETS = Path("assets")
LOGO_PATH = ASSETS / "bullvision_logo.png"   # <- put your logo image here
BG_PATH   = ASSETS / "bull_bear_bg.png"     # <- put your background image here (jpg/png)

def _set_background(img_path: Path):
    if not img_path.exists():
        return
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    ext = img_path.suffix.replace(".", "")
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(rgba(0,0,0,.55), rgba(0,0,0,.55)),
                            url("data:image/{ext};base64,{b64}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .hero {{
                max-width: 960px;
                margin: 7vh auto 4vh;
                padding: 2.2rem 2.4rem 2.0rem;
                border-radius: 20px;
                background: rgba(255,255,255,.9);
                box-shadow: 0 20px 48px rgba(0,0,0,.25);
                text-align: center;
            }}
            .hero h1 {{
                font-size: 3.2rem;
                margin: .4rem 0 .2rem;
                line-height: 1.1;
                color: #0f172a;
            }}
            .hero p.tag {{
                margin: 0;
                font-size: 1.2rem;
                color: #334155;
            }}
            .nav {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(230px,1fr));
                gap: 14px;
                margin: 1.1rem auto 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _logo_img_tag(img_path: Path, height_px=72) -> str:
    if not img_path.exists():
        return ""
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    ext = img_path.suffix.replace(".", "")
    return f'<img src="data:image/{ext};base64,{b64}" alt="BullVision" style="height:{height_px}px;margin-bottom:10px;">'

_set_background(BG_PATH)

# ---------- Hero ----------
st.markdown(
    f"""
    <div class="hero">
        {_logo_img_tag(LOGO_PATH)}
        <h1>BullVision</h1>
        <p class="tag">Shaping strong investment strategies</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Navigation ----------
def _go(page: str):
    # Works on Streamlit ≥1.30; falls back to a link otherwise
    try:
        st.switch_page(page)
    except Exception:
        st.page_link(page, label="Open →")

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("📈 Market Data Scraper", use_container_width=True):
        _go("streamlit_app.py")
    if c2.button("🧪 Feature Engineering", use_container_width=True):
        _go("pages/02_Feature_Engineering.py")
    if c3.button("🔮 Forecast", use_container_width=True):
        _go("pages/03_Forecast.py")
    if c4.button("🧭 Regime • 20% Rule", use_container_width=True):
        _go("pages/04_Regime_20_Rule.py")

st.caption("Tip: if the background doesn’t show, make sure your images are in `assets/` with the exact filenames above.")
