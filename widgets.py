"""Reusable UI pieces: color-coded answer buttons, colored dropdowns, badges.

Every answer button gets its own CSS class from Streamlit (st-key-<key>), so we
color each one individually: a tinted outline when unselected and a solid fill
(plus a check mark) when selected. The page's CSS is collected while the page
renders and injected once at the end (flush_css).
"""

import html

import streamlit as st

from data import BLUE, opts

ERR = "#f2abab"


# ---------------------------------------------------------------------------
# CSS helpers
# ---------------------------------------------------------------------------
def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def tint(hex_color, amount=0.45):
    """Mix a color with white so it reads as text on the dark background."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(c + (255 - c) * amount)  # noqa: E731
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def begin_page():
    st.session_state["_css"] = []


def add_css(rule):
    st.session_state.setdefault("_css", []).append(rule)


def flush_css():
    css = "".join(st.session_state.get("_css", []))
    if css:
        st.html(f"<style>{css}</style>")


def safe_key(*parts):
    """Streamlit keys become CSS class names, so keep them to [a-z0-9_]."""
    raw = "_".join(str(p) for p in parts).lower()
    return "".join(ch if ch.isalnum() else "_" for ch in raw)


def button_css(key, color, selected, height=44):
    sel = f".st-key-{key} button"
    if selected:
        return (
            f"{sel}{{background:{color};border:1px solid {color};color:#0f1318;"
            f"min-height:{height}px;border-radius:8px;box-shadow:0 0 0 2px {rgba(color, .30)};}}"
            f"{sel} p,{sel} span{{color:#0f1318;font-weight:650;}}"
            f"{sel}:hover,{sel}:focus{{background:{color};border-color:{color};color:#0f1318;}}"
        )
    return (
        f"{sel}{{background:{rgba(color, .10)};border:1px solid {rgba(color, .50)};"
        f"color:{tint(color)};min-height:{height}px;border-radius:8px;}}"
        f"{sel} p{{color:{tint(color)};}}"
        f"{sel}:hover{{background:{rgba(color, .24)};border-color:{color};color:#ffffff;}}"
        f"{sel}:hover p{{color:#ffffff;}}"
    )


# ---------------------------------------------------------------------------
# Labels, errors, badges
# ---------------------------------------------------------------------------
def q_label(text, required=False, hint=None):
    star = " <span class='req' aria-label='required'>*</span>" if required else ""
    hint_html = f" <span class='q-hint'>{html.escape(hint)}</span>" if hint else ""
    st.markdown(f"<div class='q-label'>{html.escape(text)}{star}{hint_html}</div>",
                unsafe_allow_html=True)


def q_error(text):
    st.markdown(f"<div class='q-err'>⚠ {html.escape(text)}</div>", unsafe_allow_html=True)


def badge(text, color=BLUE, solid=False):
    text = html.escape(str(text))
    if solid:
        return (f"<span class='badge' style='background:{color};color:#0f1318;"
                f"border:1px solid {color}'>{text}</span>")
    return (f"<span class='badge' style='background:{rgba(color, .14)};color:{tint(color, .35)};"
            f"border:1px solid {rgba(color, .45)}'>{text}</span>")


# ---------------------------------------------------------------------------
# Color-coded answer buttons
# ---------------------------------------------------------------------------
def _pick(key, value, multi, exclusive, allow_clear):
    if multi:
        cur = list(st.session_state.get(key) or [])
        if value in cur:
            cur.remove(value)
        else:
            if exclusive is not None:
                cur = [] if value == exclusive else [x for x in cur if x != exclusive]
            cur.append(value)
        st.session_state[key] = cur
    else:
        if st.session_state.get(key) == value and allow_clear:
            st.session_state[key] = None
        else:
            st.session_state[key] = value


def choice(label, key, spec, *, required=False, multi=False, exclusive=None, hint=None,
           error=None, allow_clear=True, show_label=True):
    """A row of colored buttons. Single choice by default; multi=True for checklists.

    spec: list of values or (value, color[, label]) tuples, see data.opts().
    exclusive: in multi mode, a value that clears the others (e.g. "None observed").
    """
    if show_label:
        q_label(label, required, hint)
    cur = st.session_state.get(key, [] if multi else None)
    with st.container(horizontal=True, gap="small", key=safe_key("grp", key)):
        for i, (value, color, text) in enumerate(opts(spec)):
            selected = (value in (cur or [])) if multi else (value == cur)
            bkey = safe_key("opt", key, i)
            st.button(text, key=bkey, on_click=_pick,
                      args=(key, value, multi, exclusive, allow_clear and not required),
                      icon=":material/check:" if selected else None)
            add_css(button_css(bkey, color, selected))
    if error:
        q_error(error)


def colored_select(label, key, options, *, colors=None, required=False, error=None,
                   placeholder="Choose…", format_func=str):
    """A dropdown whose border and fill take the color of the chosen answer."""
    star = " :blue[\\*]" if required else ""
    st.selectbox(label + star, options, index=None, key=key, placeholder=placeholder,
                 format_func=format_func, persist_state="session")
    val = st.session_state.get(key)
    if val is not None:
        c = (colors or {}).get(val, BLUE)
        add_css(f".st-key-{key} [data-baseweb='select'] > div{{border-color:{c} !important;"
                f"background:{rgba(c, .14)} !important;}}")
    elif error:
        add_css(f".st-key-{key} [data-baseweb='select'] > div{{border-color:{ERR} !important;}}")
    if error and val is None:
        q_error(error)


def color_dot_legend(items):
    """Small inline legend: [(text, color), ...]."""
    parts = "".join(
        f"<span class='legend-item'><span class='legend-dot' style='background:{c}'></span>"
        f"{html.escape(t)}</span>" for t, c in items)
    st.markdown(f"<div class='legend'>{parts}</div>", unsafe_allow_html=True)


GLOBAL_CSS = f"""
.block-container {{ max-width: 1240px; padding-top: 4.2rem; padding-bottom: 3rem; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
h1 {{ font-size: 1.85rem !important; }}
.brand {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px; margin-bottom: 4px; }}
.brand .logo {{ font-family: 'JetBrains Mono', Consolas, monospace; font-weight:700; color:#4a9bd1; font-size:1.15rem; }}
.brand .name {{ font-family: 'JetBrains Mono', Consolas, monospace; letter-spacing:.1em; font-size:.85rem; color:#d3dae2; }}
.brand .class {{ margin-left:auto; background:#1f5130; color:#bfe8cd; padding:4px 10px; border-radius:4px;
  font-family:'JetBrains Mono', Consolas, monospace; font-size:.75rem; font-weight:700; letter-spacing:.04em; }}
.q-label {{ font-weight: 600; font-size: 0.98rem; color:#e6ebf0; margin: 0.4rem 0 0.15rem; }}
.q-hint {{ font-weight: 400; font-size: 0.85rem; color:#a3afbb; }}
.req {{ color:#7dbbe6; font-weight:700; }}
.q-err {{ color:{ERR}; font-size: 0.88rem; margin-top: 2px; }}
.badge {{ display:inline-block; padding:2px 10px; border-radius:11px; font-size:.84rem; font-weight:600;
  margin:2px 4px 2px 0; white-space:nowrap; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:.85rem; color:#c3ccd5; margin:2px 0 8px; }}
.legend-item {{ display:flex; align-items:center; gap:6px; }}
.legend-dot {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
.kv {{ display:grid; grid-template-columns: 150px 1fr; gap:6px 12px; font-size:.93rem; margin-bottom:10px; }}
.kv .k {{ color:#a3afbb; }}
.sec-title {{ font-family:'JetBrains Mono', Consolas, monospace; font-size:.75rem; letter-spacing:.08em;
  color:#a3afbb; margin:12px 0 6px; }}
.caveat {{ display:flex; gap:10px; padding:10px 14px; border-radius:10px; background:#2a2315;
  border:1px solid #6b5527; color:#ecd39a; font-size:.92rem; line-height:1.45; margin-bottom: 8px; }}
[class*="st-key-card_"] {{ background:#171c23; border-radius:12px; padding: 18px 20px 16px; }}
[class*="st-key-step_"] button {{ min-height: 52px; border-radius: 8px; }}
[class*="st-key-opt_pc_"][class*="_rating_"] button {{ min-width: 54px; }}
"""


def page_header(title, subtitle=None):
    add_css(GLOBAL_CSS)
    st.html(
        "<div class='brand'><span class='logo'>BLP</span><span class='name'>FIELD REPORTING</span>"
        "<span class='class'>UNCLASSIFIED // PROTOTYPE · SAMPLE DATA ONLY</span></div>")
    st.title(title)
    if subtitle:
        st.caption(subtitle)

