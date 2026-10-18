import html
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import streamlit as st

BASE = Path(__file__).parent

# Paste any image URL here. Leave "" for the built-in gradient.
BG_URL = ""

PER_PAGE = 5
PAGES = 8                      # 8 pages x 5 = 40 recommendations
TOTAL = PER_PAGE * PAGES

PLACEHOLDER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='225' height='320'>"
    "<rect width='100%' height='100%' fill='%23F5D3DD'/>"
    "<text x='50%' y='50%' fill='%230D111A' font-size='18' font-family='serif' "
    "text-anchor='middle'>No Poster</text></svg>"
)

st.set_page_config(page_title="Anime Recommender", page_icon="🌸", layout="wide")

# ---------- Styling ----------
# Palette: #0D111A navy | #F9F9F9 off-white | #F5D3DD pale pink | #ED7BAB rose | #E42278 hot pink
OVERLAY = "linear-gradient(180deg, rgba(13,17,26,.82), rgba(13,17,26,.90))"
FALLBACK = (
    "radial-gradient(circle at 12% 8%, rgba(228,34,120,.28), transparent 45%), "
    "radial-gradient(circle at 92% 92%, rgba(237,123,171,.20), transparent 45%), "
    "#0D111A"
)
bg_layer = f"{OVERLAY}, url('{BG_URL}') center/cover fixed" if BG_URL else FALLBACK

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: __BG__; background-color: #0D111A; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 1400px; }

/* No grey-out flicker while Streamlit reruns */
[data-stale="true"] { opacity: 1 !important; transition: none !important; }

/* Soft glowing blobs */
.stApp::before, .stApp::after {
    content: ""; position: fixed; border-radius: 50%; filter: blur(100px);
    z-index: 0; pointer-events: none; opacity: .28;
}
.stApp::before { width: 420px; height: 420px; background: #E42278; top: -140px; left: -140px; }
.stApp::after  { width: 480px; height: 480px; background: #ED7BAB; bottom: -180px; right: -150px; }

/* Hero */
.hero { text-align: center; margin-bottom: 2rem; position: relative; z-index: 1; }
.hero .eyebrow {
    letter-spacing: .35em; text-transform: uppercase; font-size: .72rem;
    color: #ED7BAB; margin-bottom: .6rem; font-weight: 600;
}
.hero h1 {
    font-family: 'Playfair Display', serif; font-weight: 700;
    font-size: 3.4rem; margin: 0; color: #F9F9F9; padding: 0; width: 100%;
    text-shadow: 0 4px 28px rgba(228,34,120,.35);
}
.hero p { color: #F5D3DD; font-size: 1.05rem; margin-top: .5rem; font-weight: 400; }

/* Glass panel holding the widgets */
.st-key-controls {
    background: rgba(249,249,249,.06);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(245,211,221,.20); border-radius: 22px;
    padding: 1.6rem 1.8rem 1.4rem;
    box-shadow: 0 12px 40px rgba(0,0,0,.35);
}

/* Select box */
label p { color: #F5D3DD !important; font-weight: 500; letter-spacing: .02em; }
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
    background: rgba(249,249,249,.07) !important;
    background-color: rgba(249,249,249,.07) !important;
    border-color: rgba(245,211,221,.28) !important;
    border-radius: 14px !important;
    color: #F9F9F9 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: #ED7BAB !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] * { color: #F9F9F9 !important; }
[data-testid="stSelectbox"] svg { fill: #F5D3DD !important; }

/* Dropdown list (renders in a popover) */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] ul,
[data-baseweb="popover"] [role="listbox"],
[data-testid="stSelectboxVirtualDropdown"] {
    background: #151a28 !important;
    background-color: #151a28 !important;
    border-radius: 14px !important;
}
[data-baseweb="popover"] li,
[data-baseweb="popover"] li * { background: transparent !important; color: #F9F9F9 !important; }
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] li[aria-selected="true"] { background: rgba(228,34,120,.35) !important; }

/* Main button */
div.stButton { display: flex; justify-content: center; margin-top: .6rem; }
div.stButton > button {
    background: linear-gradient(90deg, #ED7BAB, #E42278);
    color: #F9F9F9; border: none; border-radius: 999px;
    padding: .6rem 2.4rem; font-weight: 600; letter-spacing: .02em;
    box-shadow: 0 8px 24px rgba(228,34,120,.40);
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px); filter: brightness(1.08); color: #F9F9F9; border: none;
    box-shadow: 0 12px 30px rgba(228,34,120,.55);
}
div.stButton > button:active { transform: scale(.98); color: #F9F9F9; }

/* Results */
.section-title {
    text-align: center; font-family: 'Playfair Display', serif;
    color: #F9F9F9; font-size: 1.7rem; margin: 2.6rem 0 .3rem; position: relative; z-index: 1;
}
.section-sub { text-align: center; color: #F5D3DD; font-size: .9rem; margin-bottom: 1.6rem; opacity: .85; }
.grid {
    display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 1.2rem; position: relative; z-index: 1;
}

/* Page transitions: the whole row slides in from the direction of travel */
.grid.fresh { animation: fresh .5s cubic-bezier(.22,.8,.3,1) both; }
.grid.next  { animation: slideNext .45s cubic-bezier(.22,.8,.3,1) both; }
.grid.prev  { animation: slidePrev .45s cubic-bezier(.22,.8,.3,1) both; }
@keyframes fresh     { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
@keyframes slideNext { from { opacity: 0; transform: translateX(48px); } to { opacity: 1; transform: none; } }
@keyframes slidePrev { from { opacity: 0; transform: translateX(-48px); } to { opacity: 1; transform: none; } }

.card {
    position: relative; border-radius: 20px; overflow: hidden;
    background: rgba(249,249,249,.06);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(245,211,221,.20);
    box-shadow: 0 10px 30px rgba(0,0,0,.45);
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
.card:hover {
    transform: translateY(-8px); border-color: rgba(237,123,171,.7);
    box-shadow: 0 20px 44px rgba(228,34,120,.28);
}
.card img { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; transition: transform .4s ease; }
.card:hover img { transform: scale(1.05); }
.card .imgwrap { overflow: hidden; position: relative; }
.rank {
    position: absolute; top: 10px; left: 10px; z-index: 2;
    width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center;
    background: rgba(13,17,26,.80); color: #F9F9F9; font-size: .8rem; font-weight: 600;
    border: 1px solid rgba(237,123,171,.6); backdrop-filter: blur(4px);
}
.card .info { padding: .85rem 1rem 1.05rem; }
.card .title {
    color: #F9F9F9; font-weight: 600; font-size: .85rem; line-height: 1.3;
    height: 2.6em; overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.bar { height: 6px; border-radius: 99px; background: rgba(245,211,221,.18); margin-top: .7rem; overflow: hidden; }
.bar span { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #ED7BAB, #E42278); }
.match { margin-top: .4rem; font-size: .75rem; color: #F5D3DD; font-weight: 500; }

/* ---- Circular arrow buttons ---- */
.st-key-prev, .st-key-next { display: flex; justify-content: center; align-items: center; }
.st-key-prev div.stButton, .st-key-next div.stButton { margin-top: 0; }
.st-key-prev button, .st-key-next button {
    width: 48px !important; min-width: 48px !important; max-width: 48px !important;
    height: 48px !important; min-height: 48px !important;
    aspect-ratio: 1 / 1 !important; flex: 0 0 48px !important;
    padding: 0 !important; border-radius: 50% !important;
    display: flex !important; align-items: center; justify-content: center;
    background: rgba(249,249,249,.08) !important;
    border: 1px solid rgba(245,211,221,.35) !important;
    box-shadow: none !important; color: #F9F9F9 !important;
    transition: background .2s ease, border-color .2s ease, transform .2s ease !important;
}
.st-key-prev button p, .st-key-next button p { margin: 0 !important; line-height: 1 !important; font-size: 1.15rem; }
.st-key-prev button:hover:not(:disabled),
.st-key-next button:hover:not(:disabled) {
    background: #E42278 !important; border-color: #E42278 !important; transform: scale(1.08) !important;
}
.st-key-prev button:disabled, .st-key-next button:disabled { opacity: .25; cursor: default; }

/* ---- Page dots ---- */
.dots { display: flex; justify-content: center; gap: .5rem; margin-top: 1.4rem; position: relative; z-index: 1; }
.dots span { width: 8px; height: 8px; border-radius: 50%; background: rgba(245,211,221,.25); transition: all .3s ease; }
.dots span.on { width: 26px; border-radius: 99px; background: #E42278; }

@media (max-width: 900px) {
    .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""
st.markdown(CSS.replace("__BG__", bg_layer), unsafe_allow_html=True)

# ---------- Data ----------
@st.cache_resource
def load_data():
    meta = pd.read_pickle(BASE / 'anime_meta.pkl')
    X = sp.load_npz(BASE / 'features.npz').tocsr()
    return meta, X

@st.cache_data(show_spinner="Finding your matches...")
def _fetch_posters_anilist(anime_ids):
    # One request for all ids. Raises on failure, so failures are NOT cached.
    query = '''
    query ($ids: [Int]) {
      Page(perPage: 50) {
        media(idMal_in: $ids, type: ANIME) { idMal coverImage { large } }
      }
    }'''
    r = requests.post(
        'https://graphql.anilist.co',
        json={'query': query, 'variables': {'ids': list(anime_ids)}},
        timeout=6
    )
    r.raise_for_status()
    media = r.json()['data']['Page']['media']
    return {m['idMal']: m['coverImage']['large'] for m in media}

def fetch_posters(anime_ids):
    ids = tuple(int(i) for i in anime_ids)
    try:
        found = _fetch_posters_anilist(ids)
    except Exception:
        found = {}
    return [found.get(i, PLACEHOLDER) for i in ids]

@st.cache_data(show_spinner=False)
def ranked(anime_name):
    """Top TOTAL anime (best first, excluding the selected one) with their scores."""
    anime_index = anime[anime['Name'] == anime_name].index[0]
    scores = (X @ X[anime_index].T).toarray().ravel()

    k = min(TOTAL + 1, len(scores) - 1)
    top = np.argpartition(-scores, k)[:k + 1]
    top = top[np.argsort(-scores[top])]
    top = [i for i in top if i != anime_index][:TOTAL]
    return top, scores

def page_items(anime_name, page):
    top, scores = ranked(anime_name)
    # Fetch all 40 posters in ONE request; later pages are then instant
    all_posters = fetch_posters(anime.iloc[top]['MAL_ID'].tolist())

    lo, hi = page * PER_PAGE, (page + 1) * PER_PAGE
    idx = top[lo:hi]
    names = anime.iloc[idx]['Name'].tolist()
    posters = all_posters[lo:hi]
    matches = [float(scores[i]) for i in idx]
    return names, posters, matches

def render_cards(names, posters, matches, page, direction):
    cards = []
    for n, (name, poster, m) in enumerate(zip(names, posters, matches), start=1):
        rank = page * PER_PAGE + n
        safe = html.escape(str(name))
        pct = max(0.0, min(m, 1.0))
        cards.append(
            f'<div class="card"><div class="imgwrap"><div class="rank">{rank}</div>'
            f'<img src="{poster}" alt="{safe}"></div>'
            f'<div class="info"><div class="title">{safe}</div>'
            f'<div class="bar"><span style="width:{pct*100:.0f}%"></span></div>'
            f'<div class="match">{pct:.0%} match</div></div></div>'
        )
    st.markdown(f'<div class="grid {direction}">{"".join(cards)}</div>', unsafe_allow_html=True)

# ---------- State / callbacks ----------
anime, X = load_data()
all_names = anime['Name'].values

st.session_state.setdefault('page', 0)
st.session_state.setdefault('direction', 'fresh')

def show_results():
    st.session_state.show = True
    st.session_state.page = 0
    st.session_state.direction = 'fresh'

def reset_page():
    st.session_state.page = 0
    st.session_state.direction = 'fresh'

def prev_page():
    st.session_state.page = max(0, st.session_state.page - 1)
    st.session_state.direction = 'prev'

def next_page():
    st.session_state.page = min(PAGES - 1, st.session_state.page + 1)
    st.session_state.direction = 'next'

# ---------- UI ----------
st.markdown(
    '<div class="hero"><div class="eyebrow">Discover · Watch · Love</div>'
    '<h1>Anime Recommender</h1>'
    '<p>Pick a series you adore and find what to watch next 🌸</p></div>',
    unsafe_allow_html=True,
)

_, mid, _ = st.columns([1, 2.4, 1])
with mid:
    with st.container(key="controls"):
        st.selectbox('Choose an anime', all_names, key='selected', on_change=reset_page)
        st.button('Recommend', on_click=show_results)

if st.session_state.get('show'):
    page = st.session_state.page
    names, posters, matches = page_items(st.session_state.selected, page)

    st.markdown(
        f'<div class="section-title">Because you liked {html.escape(st.session_state.selected)}</div>'
        f'<div class="section-sub">Ranked by similarity in genre, story, studio and format</div>',
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 12, 1], vertical_alignment="center")
    with left:
        st.button('❮', key='prev', on_click=prev_page, disabled=(page == 0))
    with center:
        render_cards(names, posters, matches, page, st.session_state.direction)
    with right:
        st.button('❯', key='next', on_click=next_page, disabled=(page == PAGES - 1))

    dots = ''.join(f'<span class="{"on" if i == page else ""}"></span>' for i in range(PAGES))
    st.markdown(f'<div class="dots">{dots}</div>', unsafe_allow_html=True)