"""Look & feel.  Palette follows a colour-blind-checked categorical order (blue, orange, aqua, yellow, magenta,
green, violet, red).  Gains/losses use a red↔teal diverging pair and are always also printed as +/- numbers."""
from __future__ import annotations

INK = "#14202B"
MUTED = "#5B6B7A"
GRID = "#E6EBF0"
SURFACE = "#FFFFFF"
BG = "#F5F7FA"
CATEGORICAL = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6250D6", "#E34948"]
NEG, MID, POS = "#C23B31", "#F2F2EE", "#0F7B6C"
DIVERGING = [NEG, MID, POS]
NAVY = "#0F2A43"
ACCENT = "#1F5FA8"

CSS = f"""
<style>
:root {{ --ink:{INK}; --muted:{MUTED}; --line:#E3E8EE; --card:#FFFFFF; --accent:{ACCENT}; --navy:{NAVY}; }}
html, body, [class*="css"] {{ color: var(--ink); }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
h1 {{ font-weight: 700; letter-spacing: -0.02em; color: var(--navy); font-size: 1.9rem !important; margin-bottom: .1rem; }}
h2, h3 {{ color: var(--navy); letter-spacing: -0.01em; }}
[data-testid="stMetric"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(15,42,67,.04); }}
[data-testid="stMetricLabel"] p {{ font-size: .80rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
[data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 650; color: var(--navy); }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
div[data-baseweb="tab-list"] {{ gap: 4px; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
.wb-sub {{ color: var(--muted); margin: 0 0 1.1rem 0; font-size: 1.0rem; }}
.wb-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; height: 100%;
    box-shadow: 0 1px 3px rgba(15,42,67,.05); }}
.wb-card h4 {{ margin: 0 0 6px 0; color: var(--navy); }}
.wb-pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; }}
.wb-ok {{ background:#E3F4EC; color:#0B6B4E; }} .wb-warn {{ background:#FFF1D6; color:#8A5A00; }} .wb-bad {{ background:#FBE3E1; color:#9C2B22; }}
.wb-note {{ border-left: 4px solid var(--accent); background:#F1F6FC; padding: 10px 14px; border-radius: 6px; margin: .4rem 0 1rem 0; color:#1c3550; }}
.wb-caution {{ border-left: 4px solid #D9922B; background:#FFF8EC; padding: 10px 14px; border-radius: 6px; margin: .4rem 0 1rem 0; color:#5b3b05; }}
.wb-verdict-good {{ border-left: 4px solid {POS}; background:#EAF6F3; padding:12px 16px; border-radius:6px; }}
.wb-verdict-neutral {{ border-left: 4px solid #8896A5; background:#F3F5F8; padding:12px 16px; border-radius:6px; }}
.wb-verdict-bad {{ border-left: 4px solid {NEG}; background:#FBEDEB; padding:12px 16px; border-radius:6px; }}
#MainMenu {{ visibility: hidden; }} footer {{ visibility: hidden; }}
</style>
"""
