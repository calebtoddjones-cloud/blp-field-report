"""
Battlefield Learning Platform - Field Report web app (Streamlit prototype)

Level 1 ("Tactical") report from the desktop app FieldReport/field_report_app_v3.py:
same questions, same database layout, so one field_reports.db works in both.

Run locally:   pip install -r requirements.txt
               streamlit run app.py
Deploy:        see README.md (Streamlit Community Cloud).

The platform catalog and all sample data are ILLUSTRATIVE, not real.
"""

import hashlib
import html
import io
import json
import os
import re
import threading
from pathlib import Path
from urllib.parse import urlsplit
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import data as D
import graphs
from i18n import language_switch, t
from widgets import (add_css, badge, begin_page, button_css, choice, color_dot_legend,
                     colored_select, flush_css, page_header, q_error, q_label, rgba, tint)

st.set_page_config(page_title="BLP Field Reporting", page_icon=":material/assignment:",
                   layout="wide")

_WRITE_LOCK = threading.Lock()


def setting(name, default=""):
    """From Streamlit secrets (Cloud: app settings > Secrets) or an environment variable."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file: fine when running locally
        pass
    return os.environ.get(name, default)


ADMIN_CODE = setting("BLP_ADMIN_CODE")           # set on Cloud so only you can delete / load data
AUTO_SAMPLE = setting("BLP_AUTO_SAMPLE", "0").lower() in ("1", "true", "yes")


@st.cache_resource
def get_conn():
    c = D.connect()
    # Cloud's disk is wiped on restart: refill with sample data so the demo always has charts
    if AUTO_SAMPLE and c.execute("SELECT COUNT(*) FROM reports WHERE tier = 1").fetchone()[0] == 0:
        D.generate_sample_data(c, seed=7)
    return c


def is_admin():
    return not ADMIN_CODE or st.session_state.get("admin_ok", False)


def admin_unlock():
    """Passcode box shown only when an admin code is set and not yet entered."""
    code = st.text_input("Organizer passcode", type="password", key="admin_try",
                         help="Deleting and loading data is limited to the organizer.")
    if code:
        if code == ADMIN_CODE:
            st.session_state["admin_ok"] = True
            st.rerun()
        else:
            st.caption(":red[Wrong passcode.]")


def app_base_url():
    try:
        u = urlsplit(st.context.url)
        return f"{u.scheme}://{u.netloc}/"
    except Exception:
        return ""


@st.dialog("Scan to open the report form", width="large")
def share_dialog():
    import qrcode   # draws with Pillow, which Streamlit already installs
    url = setting("BLP_PUBLIC_URL") or app_base_url()
    if not url:
        st.write("Couldn't work out this app's address.")
        return
    img = qrcode.make(url, box_size=16, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _l, mid, _r = st.columns([1, 2, 1])
    with mid:
        st.image(buf.getvalue(), width="stretch")
    st.markdown(f"<div style='text-align:center;font-size:1.25rem;font-weight:600;margin-top:8px'>"
                f"{html.escape(url)}</div>", unsafe_allow_html=True)
    if "localhost" in url or "127.0.0.1" in url:
        st.warning("This is a localhost address, so it only works on this computer. Share the "
                   "Streamlit Cloud link instead.")


def conn():
    return get_conn()


# ===========================================================================
# NEW REPORT
# ===========================================================================
STEPS = ["Unit & task", "Conditions", "Platforms & ratings", "Malfunctions & result",
         "Review & submit"]

FIELD_DEFAULTS = {
    "f_report_date": date.today, "f_unit_type": None, "f_unit_id": "", "f_task": None,
    "f_location": None, "f_weather": None, "f_time_of_day": None, "f_terrain": None,
    "f_ew_level": None, "f_adv_on": False,
    "f_enemy_type": "Unknown", "f_enemy_strength": "Unknown", "f_ad_level": "Unknown",
    "f_enemy_adaptation": "Unknown",
    "f_platforms": list, "f_malfunctions": list, "f_malfunction_impact": None,
    "f_outcome": None, "f_outcome_driver": None, "f_notes": "",
}


def init_form(force=False):
    if force:
        for k in [k for k in st.session_state if str(k).startswith("pc_")]:
            del st.session_state[k]
    for k, d in FIELD_DEFAULTS.items():
        if force or k not in st.session_state:
            st.session_state[k] = d() if callable(d) else d
    if force or "step" not in st.session_state:
        st.session_state["step"] = 1
    if force:
        st.session_state["show_errors"] = False


def card_key(name, field):
    """Session key for one platform card's answer. Hashing keeps keys CSS-safe for any name."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"pc_{h}_{field}"


def selected_platforms():
    return list(st.session_state.get("f_platforms") or [])


def missing_items():
    """[(id, label, step)] for every required answer still empty."""
    s = st.session_state
    out = []
    if not s.get("f_report_date"):
        out.append(("report_date", t("Report date"), 1))
    if not s.get("f_unit_type"):
        out.append(("unit_type", t("Unit type"), 1))
    if not s.get("f_task"):
        out.append(("task", t("Primary task"), 1))
    plats = selected_platforms()
    if not plats:
        out.append(("platforms", t("At least one platform"), 3))
    for p in plats:
        if not s.get(card_key(p, "effectiveness")):
            out.append((f"eff::{p}", t("Effectiveness for {p}", p=p), 3))
    if not s.get("f_malfunctions"):
        out.append(("malfunctions", t("Malfunction check"), 4))
    if not s.get("f_outcome"):
        out.append(("outcome", t("Mission outcome"), 4))
    return out


def required_total():
    return 6 + max(1, len(selected_platforms()))


def err(item_id, msg=None):
    if st.session_state.get("show_errors") and item_id in {i for i, *_ in missing_items()}:
        return msg or "Required."
    return None


def has_malfunction():
    return any(m != D.MAL_NONE for m in st.session_state.get("f_malfunctions") or [])


def goto(step):
    st.session_state["step"] = step


def reset_form():
    init_form(force=True)
    st.session_state.pop("last_submitted", None)


def build_record():
    s = st.session_state
    order = D.values(D.MALFUNCTIONS)
    mals = sorted(s["f_malfunctions"], key=order.index)
    bad = has_malfunction()
    adv = bool(s.get("f_adv_on"))
    plat_rows = []
    for p in selected_platforms():
        row = {"platform": p, "platform_status": s.get(card_key(p, "platform_status")),
               "field_modified": int(s.get(card_key(p, "field_modified")) or 0)}
        for k, *_ in D.PLATFORM_RATINGS:
            row[k] = s.get(card_key(p, k))
        plat_rows.append(row)
    rec = {
        "report_date": s["f_report_date"].isoformat(), "unit_type": s["f_unit_type"],
        "unit_id": (s["f_unit_id"] or "").strip() or None, "task": s["f_task"],
        "location": s["f_location"], "weather": s["f_weather"], "time_of_day": s["f_time_of_day"],
        "terrain": s["f_terrain"], "ew_level": s["f_ew_level"],
        **{k: (s[f"f_{k}"] if adv else None) for k in D.ADVERSARY_KEYS},
        "malfunctions": "; ".join(mals),
        "malfunction_impact": s["f_malfunction_impact"] if bad else "No impact",
        "outcome": s["f_outcome"],
        "outcome_driver": s["f_outcome_driver"] if s["f_outcome"] != "Success" else None,
        "notes": (s["f_notes"] or "").strip() or None,
        "tier": D.MODEL_TIER, "tier_answers": None,
        **D.rollup(plat_rows),
    }
    return rec, plat_rows


def submit_report():
    if missing_items():
        st.session_state["show_errors"] = True
        return
    rec, plat_rows = build_record()
    with _WRITE_LOCK:
        rid = D.insert_report(conn(), rec, plat_rows)
    init_form(force=True)
    st.session_state["last_submitted"] = rid
    st.session_state["reported"] = True          # unlocks Analyst login on the start page (per browser session)


def platform_meta(name):
    return D.PLATFORM_INFO.get(name, ("Unlisted / custom", "Custom"))


def platform_label(name):
    cat, origin = platform_meta(name)
    return f"{name}  ·  {t(cat)}  ·  {t(origin)}"


ORIGIN_COLORS = {"U.S.": D.BLUE, "Ukraine": D.YELLOW, "Other partner": D.GRAY, "Custom": D.ORANGE}


# ----- form sections --------------------------------------------------------
def section_unit():
    with st.container(key="card_unit"):
        st.subheader(t("Unit & task"))
        c1, c2 = st.columns(2)
        with c1:
            st.date_input(t("Report date") + " :blue[\\*]", key="f_report_date", format="YYYY-MM-DD",
                          persist_state="session")
            colored_select("Unit type", "f_unit_type", D.UNIT_TYPES, required=True,
                           error=err("unit_type", "Required: unit type."))
        with c2:
            st.text_input(t("Unit ID (optional)"), key="f_unit_id", placeholder=t("e.g. U-417"),
                          persist_state="session")
            colored_select("Primary task", "f_task", D.TASKS, required=True,
                           error=err("task", "Required: primary task."))


def section_conditions():
    with st.container(key="card_conditions"):
        st.subheader(t("Location & conditions (optional)"))
        st.caption(t("Fill in what you know. Click a selected answer again to clear it."))
        colored_select("Sector / location", "f_location", D.LOCATIONS)
        choice("Enemy EW / jamming", "f_ew_level", D.EW_LEVELS,
               hint="“None” means no jamming; leave blank if you don't know.")
        choice("Weather", "f_weather", D.WEATHER)
        c1, c2 = st.columns([1, 2])
        with c1:
            choice("Time of day", "f_time_of_day", D.TIME_OF_DAY)
        with c2:
            choice("Terrain", "f_terrain", D.TERRAIN)
    with st.container(key="card_adversary"):
        st.subheader(t("Adversary (optional)"))
        on = st.toggle(t("Add adversary details"), key="f_adv_on", persist_state="session",
                       help=t("Off = nothing recorded for the adversary (not even “Unknown”)."))
        if on:
            colored_select("Enemy type", "f_enemy_type", D.ENEMY_TYPES)
            choice("Estimated enemy strength", "f_enemy_strength", D.ENEMY_STRENGTH,
                   allow_clear=False)
            choice("Enemy air defense", "f_ad_level", D.THREAT, allow_clear=False)
            choice("Enemy adaptation", "f_enemy_adaptation", D.ENEMY_ADAPT, allow_clear=False)
        else:
            st.caption(t("Skipped. Turn on to describe the enemy force."))


def platform_picker():
    all_names = sorted(D.PLATFORM_INFO, key=lambda n: (D.PLATFORM_INFO[n][0], n))
    extra = [p for p in selected_platforms() if p not in D.PLATFORM_INFO]
    st.multiselect(
        t("Platforms employed") + " :blue[\\*]", all_names + extra, key="f_platforms",
        format_func=platform_label, accept_new_options=True, persist_state="session",
        filter_mode="contains", select_all=False,
        placeholder=t("Type a name or a category, e.g. Javelin, Puma, FPV, Anti-Armor"),
        help=t("Pick every platform you used. Not listed? Type its name and press Enter "
               "to add it as a custom platform."))
    if e := err("platforms", "Required: pick at least one platform."):
        q_error(e)


IMG_DIR = Path(__file__).parent / "assets" / "platforms"


def file_slug(text):
    """Same file-name rule as the desktop app: lower case, runs of other characters -> '_'."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def platform_image(name):
    """Picture rendered from the BLP 3D model: the platform's own, else its category's, else None."""
    cat, _ = platform_meta(name)
    for stem in (file_slug(name), file_slug(cat)):
        f = IMG_DIR / f"{stem}.png"
        if stem and f.exists():
            return f
    return None


def platform_card(name, idx, compact=False):
    cat, origin = platform_meta(name)
    img = platform_image(name)
    with st.container(key=f"card_plat_{idx}"):
        top_l, top_r = st.columns([2.6, 1], vertical_alignment="center")
        with top_r:
            if img:
                st.image(str(img), width="stretch")
                st.caption(t("BLP stylized model · illustrative"))
        with top_l:
            card_top(name, cat, origin, compact)
        if compact:
            return
        traits_block(name)


def card_top(name, cat, origin, compact):
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:4px'>"
        f"<span style='font-size:1.15rem;font-weight:650;color:#eef2f6'>{html.escape(name)}</span>"
        f"<span style='color:#a3afbb;font-size:.9rem'>{html.escape(t(cat))}</span>"
        f"{badge(origin, ORIGIN_COLORS.get(origin, D.ORANGE))}</div>",
        unsafe_allow_html=True)
    if not compact:
        choice("How did it perform?", card_key(name, "platform_status"), D.PLATFORM_STATUS)
    choice("Effectiveness (overall)", card_key(name, "effectiveness"), D.EFFECTIVENESS,
           required=True,
           error=err(f"eff::{name}", "Required: rate effectiveness from 1 to 5."))


def traits_block(name):
    st.markdown(f"<div class='sec-title'>{t('TRAITS (OPTIONAL) · 1 = POOR, 5 = BEST')}</div>",
                unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for i, (key, label, hint, _req) in enumerate(D.TRAITS):
        with cols[i % 2]:
            choice(label, card_key(name, key), D.RATING, hint=hint)
    choice("Field-modified by the unit?", card_key(name, "field_modified"),
           D.YES_NO_NEUTRAL)


def section_platforms(compact=False):
    with st.container(key="card_platforms"):
        st.subheader(t("Which platforms did you use?"))
        platform_picker()
        plats = selected_platforms()
        if plats:
            st.caption(t("{n} selected. Each one gets its own rating card below.", n=len(plats)))
        else:
            st.caption(t("Nothing selected yet."))
    for i, p in enumerate(plats):
        platform_card(p, i, compact=compact)


def section_malfunctions_result():
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(key="card_malfunctions"):
        st.subheader(t("Malfunctions (whole engagement)"))
        choice("Jams / malfunctions observed", "f_malfunctions", D.MALFUNCTIONS, required=True,
               multi=True, exclusive=D.MAL_NONE,
               hint="Pick “None observed” or every problem you saw.",
               error=err("malfunctions", "Required: pick “None observed” or a malfunction."))
        if has_malfunction():
            st.markdown(f"<div class='sec-title'>{t('▸ SHOWN BECAUSE YOU REPORTED A MALFUNCTION')}</div>",
                        unsafe_allow_html=True)
            choice("Impact on the mission", "f_malfunction_impact", D.MAL_IMPACT[1:])
    with c2, st.container(key="card_result"):
        st.subheader(t("Mission result"))
        choice("Mission outcome", "f_outcome", D.OUTCOMES, required=True,
               error=err("outcome", "Required: pick the mission outcome."))
        o = st.session_state.get("f_outcome")
        if o:
            st.markdown(f"<div class='q-hint' style='margin-top:4px'>{html.escape(t(o))}: "
                        f"{html.escape(t(D.OUTCOME_DEFS[o]))}</div>", unsafe_allow_html=True)
        if o and o != "Success":
            choice("What most limited the result? (optional)", "f_outcome_driver",
                   D.OUTCOME_DRIVERS)
    with st.container(key="card_notes"):
        st.text_area(t("Notes (optional)"), key="f_notes", persist_state="session",
                     placeholder=t("Anything that didn't fit above."))


def kv_rows(items):
    body = "".join(f"<span class='k'>{html.escape(t(k))}</span><span>{v}</span>" for k, v in items)
    return f"<div class='kv'>{body}</div>"


def b_field(field, value):
    """Badge for a stored value, colored by its question's scale."""
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return "<span style='color:#8d99a6'>—</span>"
    if isinstance(value, float):
        value = int(value)
    spec = D.FIELD_SPECS.get(field)
    if spec is None:
        return badge(value, D.BLUE)
    return badge(D.label_of(spec, value), D.color_of(spec, value))


def trait_badges(get):
    """Compact colored badges for the six traits; get(key) returns a 1-5 value or None."""
    out = []
    for k, *_ in D.TRAITS:
        v = get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out.append(badge(f"{t(D.TRAIT_SHORT[k])} –", D.GRAY))
        else:
            v = int(v)
            out.append(badge(f"{t(D.TRAIT_SHORT[k])} {v}", D.color_of(D.RATING, v)))
    return "".join(out)


def section_review():
    s = st.session_state
    miss = missing_items()
    miss_ids = {i for i, *_ in miss}
    with st.container(key="card_review"):
        st.subheader(t("Review before you submit"))
        if miss:
            st.markdown(
                "<div class='q-err' style='font-size:.95rem;margin-bottom:8px'>⚠ "
                + (t("{n} required answer(s) still missing.", n=len(miss)) if st.session_state.get("_tr_on")
                   else f"{len(miss)} required answer{'s' if len(miss) > 1 else ''} still missing.")
                + "</div>",
                unsafe_allow_html=True)
            with st.container(horizontal=True, gap="small"):
                for n, (_i, label, step) in enumerate(miss):
                    st.button(t("Fix: {s}", s=label), key=f"fix_{n}", on_click=goto, args=(step,),
                              icon=":material/error:")
                    add_css(button_css(f"fix_{n}", D.RED, False, 40))

        def req(item, value_html):
            return badge("Missing", D.RED, solid=True) if item in miss_ids else value_html

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(f"<div class='sec-title'>{t('UNIT & TASK')}</div>" + kv_rows([
                ("Date", req("report_date", html.escape(str(s["f_report_date"] or "")))),
                ("Unit", req("unit_type", b_field("unit_type", s["f_unit_type"]))),
                ("Unit ID", html.escape(s["f_unit_id"] or "—")),
                ("Task", req("task", b_field("task", s["f_task"]))),
            ]), unsafe_allow_html=True)
            st.markdown(f"<div class='sec-title'>{t('LOCATION & CONDITIONS')}</div>" + kv_rows([
                ("Sector", b_field("location", s["f_location"])),
                ("Enemy EW", b_field("ew_level", s["f_ew_level"])),
                ("Weather", b_field("weather", s["f_weather"])),
                ("Time of day", b_field("time_of_day", s["f_time_of_day"])),
                ("Terrain", b_field("terrain", s["f_terrain"])),
            ]), unsafe_allow_html=True)
            if s.get("f_adv_on"):
                adv = kv_rows([
                    ("Enemy type", b_field("enemy_type", s["f_enemy_type"])),
                    ("Strength", b_field("enemy_strength", s["f_enemy_strength"])),
                    ("Air defense", b_field("ad_level", s["f_ad_level"])),
                    ("Adaptation", b_field("enemy_adaptation", s["f_enemy_adaptation"])),
                ])
            else:
                adv = f"<div class='q-hint'>{t('Skipped (nothing recorded).')}</div>"
            st.markdown(f"<div class='sec-title'>{t('ADVERSARY')}</div>" + adv, unsafe_allow_html=True)
        with c2:
            mals = "".join(badge(m, D.color_of(D.MALFUNCTIONS, m)) for m in s["f_malfunctions"])
            res = [("Outcome", req("outcome", b_field("outcome", s["f_outcome"])))]
            if s["f_outcome"] and s["f_outcome"] != "Success":
                res.append(("Limiting factor", b_field("outcome_driver", s["f_outcome_driver"])))
            res.append(("Malfunctions", req("malfunctions", mals)))
            if has_malfunction():
                res.append(("Impact", b_field("malfunction_impact", s["f_malfunction_impact"])))
            st.markdown(f"<div class='sec-title'>{t('MALFUNCTIONS & RESULT')}</div>" + kv_rows(res),
                        unsafe_allow_html=True)
            if s["f_notes"]:
                st.markdown(f"<div class='sec-title'>{t('NOTES')}</div>", unsafe_allow_html=True)
                st.text(s["f_notes"])
        st.markdown(f"<div class='sec-title'>{t('PLATFORMS')}</div>", unsafe_allow_html=True)
        plats = selected_platforms()
        if not plats:
            st.markdown(req("platforms", ""), unsafe_allow_html=True)
        for p in plats:
            eff = s.get(card_key(p, "effectiveness"))
            st.markdown(
                f"<div style='margin:4px 0 10px'><b>{html.escape(p)}</b> "
                f"<span class='q-hint'>{html.escape(t(platform_meta(p)[0]))}</span><br>"
                + b_field("platform_status", s.get(card_key(p, "platform_status")))
                + (b_field("effectiveness", eff) if eff else badge("Effectiveness missing", D.RED, solid=True))
                + trait_badges(lambda k, p=p: s.get(card_key(p, k)))
                + (badge("Field-modified", D.BLUE) if s.get(card_key(p, "field_modified")) else "")
                + "</div>", unsafe_allow_html=True)


def stepper(step):
    miss_steps = {st_ for *_x, st_ in missing_items()}
    show_err = st.session_state.get("show_errors")
    with st.container(horizontal=True, gap="small", key="stepper"):
        for i, name in enumerate(STEPS, 1):
            if i == step:
                color, solid, icon = D.BLUE, True, None
            elif show_err and i in miss_steps:
                color, solid, icon = D.RED, False, ":material/error:"
            elif i < step and i not in miss_steps:
                color, solid, icon = D.GREEN, False, ":material/check:"
            elif i < step:
                color, solid, icon = D.ORANGE, False, ":material/more_horiz:"
            else:
                color, solid, icon = D.GRAY, False, None
            key = f"step_{i}"
            st.button(f"{i} · {t(name)}", key=key, on_click=goto, args=(i,), icon=icon,
                      width="stretch")
            add_css(button_css(key, color, solid, 52))


def progress_bar():
    total = required_total()
    done = total - len(missing_items())
    st.progress(done / total, text=t("{done} of {total} required answers done", done=done, total=total))


def page_new_report():
    init_form()
    language_switch()
    page_header("Daily engagement report",
                "Level 1 · Tactical. About 2 minutes. Only answers marked * are required.")

    if rid := st.session_state.get("last_submitted"):
        with st.container(key="card_done"):
            st.success(t("Report #{rid} sent. Thank you.", rid=rid), icon=":material/check_circle:")
            st.html(flow_html())
            with st.container(horizontal=True):
                if st.button(t("Return to start"), type="primary", icon=":material/home:", key="done_home"):
                    reset_form()
                    st.session_state["view"] = None
                    st.switch_page(PAGE_HOME)
                st.button(t("File another report"), on_click=reset_form, icon=":material/add:")
        return
    if st.session_state.get("view") != "analyst":
        if st.button(t("Back to start"), icon=":material/arrow_back:", type="tertiary", key="form_home"):
            st.session_state["view"] = None
            st.switch_page(PAGE_HOME)

    quick = st.toggle(t("Quick mode: required answers only"), key="quick", persist_state="session",
                      help=t("Shows only the required questions on one page."))

    if quick:
        progress_bar()
        with st.container(key="card_quick"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.date_input(t("Report date") + " :blue[\\*]", key="f_report_date",
                              format="YYYY-MM-DD", persist_state="session")
            with c2:
                colored_select("Unit type", "f_unit_type", D.UNIT_TYPES, required=True,
                               error=err("unit_type", "Required: unit type."))
            with c3:
                colored_select("Primary task", "f_task", D.TASKS, required=True,
                               error=err("task", "Required: primary task."))
        section_platforms(compact=True)
        with st.container(key="card_quick2"):
            choice("Jams / malfunctions observed", "f_malfunctions", D.MALFUNCTIONS,
                   required=True, multi=True, exclusive=D.MAL_NONE,
                   error=err("malfunctions", "Required: pick “None observed” or a malfunction."))
            choice("Mission outcome", "f_outcome", D.OUTCOMES, required=True,
                   error=err("outcome", "Required: pick the mission outcome."))
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button(t("Submit report"), type="primary", on_click=submit_report,
                      icon=":material/send:", key="submit_quick")
        if st.session_state.get("show_errors") and missing_items():
            st.error(t("Some required answers are missing. They're marked in red above."))
        return

    step = st.session_state["step"]
    stepper(step)
    progress_bar()

    [section_unit, section_conditions, section_platforms, section_malfunctions_result,
     section_review][step - 1]()

    st.write("")
    left, right = st.columns([1, 1], vertical_alignment="center")
    with left:
        st.caption(t("Your answers are kept while you move between steps."))
    with right, st.container(horizontal=True, horizontal_alignment="right"):
        if step > 1:
            st.button(t("Back: {s}", s=t(STEPS[step - 2])), on_click=goto, args=(step - 1,), key="nav_back",
                      icon=":material/arrow_back:")
        if step < len(STEPS):
            st.button(t("Next: {s}", s=t(STEPS[step])), type="primary", on_click=goto, args=(step + 1,),
                      key="nav_next", icon=":material/arrow_forward:", icon_position="right")
        else:
            st.button(t("Submit report"), type="primary", on_click=submit_report, key="submit",
                      icon=":material/send:")
    if step == len(STEPS) and st.session_state.get("show_errors") and missing_items():
        st.error(t("Can't submit yet. Use the red Fix buttons above to finish the missing answers."))


# ===========================================================================
# REPORT DATABASE
# ===========================================================================
SAMPLE_SIZES = [50, 150, 300, 600]


def load_sample(n=150):
    with _WRITE_LOCK:
        D.generate_sample_data(conn(), n=int(n))
    st.toast(f"Added {int(n)} synthetic level 1 reports (plus 12 level 2–4).", icon=":material/science:")


def sample_tools(where):
    """Organizer box: add N random synthetic reports so the charts can be shown growing. Real reports
    from the form keep working alongside; sample rows are marked "[sample data]" in their notes."""
    with st.expander("Demo data: generate random reports", icon=":material/science:"):
        if not is_admin():
            admin_unlock()
            return
        c1, c2, c3 = st.columns([1.2, 1.2, 3], vertical_alignment="bottom")
        n = c1.selectbox("How many", SAMPLE_SIZES, index=2, key=f"gen_n_{where}")
        c2.button(f"Generate {n}", type="primary", icon=":material/casino:", key=f"gen_go_{where}",
                  on_click=load_sample, args=(n,))
        total = int(conn().execute("SELECT COUNT(*) FROM reports WHERE tier = 1").fetchone()[0])
        n_sample = int(conn().execute("SELECT COUNT(*) FROM reports WHERE tier = 1 AND notes = '[sample data]'")
                       .fetchone()[0])
        c3.caption(f"Now: {total} level 1 reports ({n_sample} synthetic, {total - n_sample} entered on the "
                   "form). Each click adds more on top; the charts redraw with the new totals. Synthetic "
                   "reports carry the planted EW pattern.")


def empty_state():
    st.info("No level 1 reports yet. Submit one from New report"
            + (", or load synthetic sample data to try the database and insights views."
               if is_admin() else "."), icon=":material/inbox:")
    if is_admin():
        st.button("Load 300 sample reports", type="primary", on_click=load_sample, args=(300,),
                  icon=":material/science:", key="load_sample_empty")


def short_mal(m):
    return "None" if m == D.MAL_NONE else (m or "—")


def cell_style(color):
    return f"background-color:{rgba(color, .20)};color:{tint(color, .35)};font-weight:600"


def other_levels_note():
    n = D.other_level_count(conn())
    if n:
        st.caption(f"{n} level 2–4 report{'s' if n != 1 else ''} in this database "
                   f"{'are' if n != 1 else 'is'} not shown here; open them in the desktop app.")


def page_database():
    page_header("Report database")
    add_css(".block-container { max-width: 1560px; }")
    df = D.reports_df(conn())
    if df.empty:
        empty_state()
        return

    n_mal = int((df["malfunctions"] != D.MAL_NONE).sum())
    st.caption(f"{len(df)} level 1 reports · {n_mal} with malfunctions · "
               f"latest {df['report_date'].max()}")
    other_levels_note()

    with st.container(key="card_filters"):
        c = st.columns([2, 1.3, 1.3, 1.1, 1.1])
        q = c[0].text_input("Search", placeholder="Platform, unit ID or note", key="db_q")
        sector = c[1].selectbox("Sector", ["All"] + D.LOCATIONS, key="db_sector")
        outcome = c[2].selectbox("Outcome", ["All"] + D.values(D.OUTCOMES), key="db_outcome")
        ew = c[3].selectbox("Enemy EW", ["All"] + D.values(D.EW_LEVELS) + ["Not given"],
                            key="db_ew")
        mal = c[4].selectbox("Malfunctions", ["All", "Any", "None"], key="db_mal")

    view = df
    if q:
        hay = (view["platforms"].fillna("") + " " + view["unit_id"].fillna("") + " "
               + view["notes"].fillna("")).str.lower()
        view = view[hay.str.contains(q.lower(), regex=False)]
    if sector != "All":
        view = view[view["location"] == sector]
    if outcome != "All":
        view = view[view["outcome"] == outcome]
    if ew == "Not given":
        view = view[view["ew_level"].isna() | (view["ew_level"] == "")]
    elif ew != "All":
        view = view[view["ew_level"] == ew]
    if mal == "Any":
        view = view[view["malfunctions"] != D.MAL_NONE]
    elif mal == "None":
        view = view[view["malfunctions"] == D.MAL_NONE]
    view = view.reset_index(drop=True)

    table = pd.DataFrame({
        "ID": view["id"], "Date": view["report_date"], "Outcome": view["outcome"],
        "Eff.": view["effectiveness"], "Malfunctions": view["malfunctions"].map(short_mal),
        "Unit": view["unit_type"], "Task": view["task"], "Sector": view["location"].fillna("—"),
        "Platforms": view["platforms"],
    })
    styled = (table.style
              .map(lambda v: cell_style(D.color_of(D.OUTCOMES, v)), subset=["Outcome"])
              .map(lambda v: cell_style(D.color_of(D.EFFECTIVENESS, v)) if pd.notna(v) else "",
                   subset=["Eff."])
              .map(lambda v: f"color:{'#f0a0a0' if v not in ('None', '—') else '#8d99a6'}",
                   subset=["Malfunctions"]))

    left, right = st.columns([2.4, 1], gap="medium")
    with left:
        color_dot_legend([("Success", D.GREEN), ("Partial", D.YELLOW), ("Aborted", D.ORANGE),
                          ("Failure", D.RED)])
        event = st.dataframe(
            styled, hide_index=True, on_select="rerun", selection_mode="single-row",
            key="db_table", height=520,
            column_config={
                "ID": st.column_config.NumberColumn(width="small", format="%d"),
                "Date": st.column_config.TextColumn(width="small"),
                "Eff.": st.column_config.NumberColumn(
                    width="small", help="Average of the platforms' effectiveness, 1–5"),
                "Platforms": st.column_config.TextColumn(width="large"),
            })
        st.caption(f"Showing {len(view)} of {len(df)} · click a row to open it · click a "
                   "column header to sort")
    rows = event.selection.rows if event and event.selection else []
    with right:
        if rows and rows[0] < len(view):
            report_detail(view.iloc[rows[0]])
        else:
            with st.container(key="card_detail_empty"):
                st.markdown("**Select a report**")
                st.caption("Click any row to see the full report here, including each "
                           "platform's own ratings.")

    with st.expander("Manage data: export, sample data, delete"):
        if not is_admin():
            st.download_button("Export level 1 reports (CSV)", df.to_csv(index=False),
                               file_name="field_reports.csv", mime="text/csv",
                               icon=":material/download:", key="csv_public")
            admin_unlock()
            return
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Export level 1 reports (CSV)", df.to_csv(index=False),
                               file_name="field_reports.csv", mime="text/csv",
                               icon=":material/download:")
        with c2:
            n = st.selectbox("Sample reports to add", SAMPLE_SIZES, index=2, key="gen_n_db")
            st.button(f"Add {n} sample reports", on_click=load_sample, args=(n,),
                      icon=":material/science:", key="load_sample_more")
        with c3:
            with st.popover("Delete all level 1 reports", icon=":material/delete:"):
                st.write("This removes every level 1 report from this database.")
                if st.button("Yes, delete them", type="primary", key="del_all"):
                    with _WRITE_LOCK:
                        D.delete_reports(conn(), df["id"].tolist())
                    st.rerun()


def report_detail(r):
    r = r.astype(object).where(r.notna(), None)   # pandas NaN -> None, so empty answers show as blank
    rid = int(r["id"])
    with st.container(key="card_detail"):
        st.markdown(
            f"<div class='sec-title' style='margin-top:0;color:#9ccbec'>REPORT #{rid} · "
            f"{html.escape(str(r['report_date']))}</div>"
            f"<div style='font-size:1.1rem;font-weight:600;margin-bottom:6px'>"
            f"{html.escape(str(r['unit_type']))} · {html.escape(str(r['task']))}"
            f"{' · ' + html.escape(str(r['location'])) if r['location'] else ''}</div>"
            + b_field("outcome", r["outcome"])
            + (b_field("outcome_driver", r["outcome_driver"]) if r["outcome_driver"] else "")
            + badge(f"Enemy EW: {r['ew_level'] or 'not given'}", D.color_of(D.THREAT, r["ew_level"])),
            unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>PLATFORMS (EACH RATED ON ITS OWN CARD)</div>",
                    unsafe_allow_html=True)
        pdf = D.platform_rows_df(conn(), rid)
        for _, p in pdf.iterrows():
            note = "" if p["rated_individually"] else (
                "<br><span class='q-hint'>Older report: one rating copied to every platform.</span>")
            st.markdown(
                f"<div style='margin:2px 0 10px'><b>{html.escape(p['platform'])}</b> "
                f"<span class='q-hint'>{html.escape(p['category'])} · {html.escape(p['origin'])}</span><br>"
                + b_field("platform_status", p["p_status"])
                + b_field("effectiveness", p["p_effectiveness"])
                + trait_badges(lambda k, p=p: p[f"p_{k}"])
                + note + "</div>", unsafe_allow_html=True)

        mals = "".join(badge(m, D.color_of(D.MALFUNCTIONS, m))
                       for m in str(r["malfunctions"] or "").split("; ") if m)
        st.markdown("<div class='sec-title'>MALFUNCTIONS</div>" + kv_rows([
            ("Observed", mals or "—"),
            ("Impact", b_field("malfunction_impact", r["malfunction_impact"])),
        ]), unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>CONDITIONS</div>" + kv_rows([
            ("Weather", b_field("weather", r["weather"])),
            ("Time / terrain", b_field("time_of_day", r["time_of_day"]) + b_field("terrain", r["terrain"])),
        ]), unsafe_allow_html=True)
        if any(r[k] for k in D.ADVERSARY_KEYS):
            adv = kv_rows([
                ("Enemy", b_field("enemy_type", r["enemy_type"]) + b_field("enemy_strength", r["enemy_strength"])),
                ("Air defense", b_field("ad_level", r["ad_level"])),
                ("Adaptation", b_field("enemy_adaptation", r["enemy_adaptation"])),
            ])
        else:
            adv = "<div class='q-hint'>Section skipped.</div>"
        st.markdown(f"<div class='sec-title'>{t('ADVERSARY')}</div>" + adv, unsafe_allow_html=True)
        if r["notes"]:
            st.markdown(f"<div class='sec-title'>{t('NOTES')}</div>", unsafe_allow_html=True)
            st.text(r["notes"])
        if not is_admin():
            return
        with st.popover("Delete this report", icon=":material/delete:"):
            if st.button(f"Yes, delete report #{rid}", type="primary", key="del_one"):
                with _WRITE_LOCK:
                    D.delete_reports(conn(), [rid])
                st.rerun()


# ===========================================================================
# INSIGHTS
# ===========================================================================
NOT_RECORDED = "(not recorded)"
HEAT_MIN_N = 3   # same floor as the desktop heatmap: cells with fewer ratings stay empty
COMPARE = {      # same conditions as the desktop heatmap (HEAT_CONDITIONS)
    "Enemy EW / jamming": ("ew_level", D.values(D.THREAT)),
    "Time of day": ("time_of_day", D.values(D.TIME_OF_DAY)),
    "Weather": ("weather", D.values(D.WEATHER)),
    "Terrain": ("terrain", D.values(D.TERRAIN)),
    "Primary task": ("task", D.TASKS),
    "Unit type": ("unit_type", D.UNIT_TYPES),
    "Enemy type": ("enemy_type", D.ENEMY_TYPES),
    "Enemy air defense": ("ad_level", D.values(D.THREAT)),
}


ANALYST_FILE = Path(__file__).parent / "analyst_base.html"


@st.cache_resource
def analyst_base():
    """The desktop project's v3 HTML page (analyst features + baked 3D models), built by build_analyst.py."""
    return ANALYST_FILE.read_text(encoding="utf-8") if ANALYST_FILE.exists() else None


def page_insights():
    page_header("Insights", "Analyst view. Same tables, scorecard math, charts, weapon profiles, 3D models and "
                "sector map as the desktop v3 app, reading this app's live database.")
    add_css(".block-container { max-width: 100%; padding-left: 1.5rem; padding-right: 1.5rem; }")
    if D.reports_df(conn()).empty:
        empty_state()
    sample_tools("ins")
    console, quick = st.tabs(["Analyst console", "Quick summary"])
    with console:
        base = analyst_base()
        if not base:
            st.error("analyst_base.html is missing from the app folder, so the analyst console can't load.")
        else:
            snap = json.dumps(D.db_snapshot(conn()), default=str).replace("</", "<\\/")
            # data is JSON with "</" escaped, so text typed into a report can't end the script block
            st.iframe(base.replace("/*__BLP_DB__*/", "window.__BLP_DB__=" + snap + ";", 1), height=1650)
            st.caption("Read-only view of the live database. Tabs inside: Insights (Tactical, Officer, "
                       "Sector map), Scorecard (radar, condition heatmap, weapon profile, 3D) and Report "
                       "list. Click a scorecard row and press PROFILE or VIEW 3D.")
    with quick:
        quick_summary()


def quick_summary():
    df = D.reports_df(conn())
    if df.empty:
        st.caption("No level 1 reports yet.")
        return
    cards = D.platform_rows_df(conn())
    rated = cards[(cards["rated_individually"] == 1) & cards["p_effectiveness"].notna()]

    st.html(
        "<div class='caveat'><span>⚠</span><span><b>Level 1 reports only.</b> If the database "
        "holds sample data, it is <b>synthetic</b>: the generator builds the EW pattern in on "
        "purpose, so these charts test the pipeline; they are not findings. Plain averages, not "
        "adjusted for small samples (the desktop Scorecard does that).</span></div>")
    other_levels_note()

    n = len(df)
    k = st.columns(4)
    k[0].metric("Level 1 reports", n, border=True, help=f"{len(cards)} platform cards")
    k[1].metric("Success rate", f"{(df['outcome'] == 'Success').mean():.0%}", border=True)
    k[2].metric("Reports with a malfunction", f"{(df['malfunctions'] != D.MAL_NONE).mean():.0%}",
                border=True)
    k[3].metric("Platforms lost", f"{(cards['p_status'] == D.STATUS_LOST).mean():.0%}",
                border=True, help="Share of platform cards marked Lost / destroyed")

    # ----- heatmap ---------------------------------------------------------
    with st.container(key="card_heat"):
        a, b = st.columns([3, 1.2], vertical_alignment="bottom")
        with a:
            st.subheader("Average effectiveness by platform category")
            st.caption("From each platform's own card. 1 = very poor (red), 5 = excellent "
                       f"(green). Cells with fewer than {HEAT_MIN_N} ratings are left empty.")
        with b:
            dim_label = st.selectbox("Compare by", list(COMPARE), key="ins_dim")
        col, order = COMPARE[dim_label]
        h = rated.assign(cond=rated[col].fillna("").replace("", NOT_RECORDED))
        g = (h.groupby(["category", "cond"])["p_effectiveness"].agg(avg="mean", n="size")
             .reset_index())
        g = g[g["n"] >= HEAT_MIN_N]
        if g.empty:
            st.caption("Not enough rated platforms yet.")
        else:
            found = set(g["cond"])
            x_order = [o for o in order if o in found] + sorted(found - set(order) - {NOT_RECORDED})
            if NOT_RECORDED in found:
                x_order.append(NOT_RECORDED)
            totals = rated.groupby("category")["p_effectiveness"].mean().sort_values(ascending=False)
            cat_order = [c for c in totals.index if c in set(g["category"])]
            g["label"] = g["avg"].map(lambda v: f"{v:.1f}")
            g["nlab"] = g["n"].map(lambda v: f"n={v}")
            base = alt.Chart(g).encode(
                x=alt.X("cond:N", sort=x_order, title=None,
                        axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=12, labelLimit=140)),
                y=alt.Y("category:N", sort=cat_order, title=None,
                        axis=alt.Axis(labelFontSize=12, labelLimit=240)))
            rect = base.mark_rect(cornerRadius=5, stroke="#0f1318", strokeWidth=3).encode(
                color=alt.Color("avg:Q", scale=alt.Scale(domain=[1, 2, 3, 4, 5], range=D.SCALE_5),
                                legend=alt.Legend(title="Avg effectiveness", orient="bottom",
                                                  gradientLength=260)),
                tooltip=[alt.Tooltip("category:N", title="Category"),
                         alt.Tooltip("cond:N", title=dim_label),
                         alt.Tooltip("avg:Q", title="Avg effectiveness", format=".2f"),
                         alt.Tooltip("n:Q", title="Ratings")])
            t1 = base.mark_text(fontSize=14, fontWeight="bold", dy=-7, color="#0f1318").encode(
                text="label:N")
            t2 = base.mark_text(fontSize=10, dy=10, color="#0f1318").encode(text="nlab:N")
            st.altair_chart((rect + t1 + t2).properties(height=alt.Step(46)), width="stretch")

    # ----- platform traits by category -----------------------------------------
    with st.container(key="card_cats"):
        st.subheader("Platform traits by category")
        st.caption("Averages from each platform's own card, 1–5, higher is better. "
                   "Loss = share of cards marked Lost / destroyed.")
        agg = {"Effectiveness": ("p_effectiveness", "mean"), "Cards": ("p_effectiveness", "size")}
        agg.update({D.TRAIT_SHORT[k]: (f"p_{k}", "mean") for k, *_ in D.TRAITS})
        t = rated.assign(lost=(rated["p_status"] == D.STATUS_LOST).astype(float))
        agg["Loss"] = ("lost", "mean")
        summ = (t.groupby("category").agg(**agg).sort_values("Effectiveness", ascending=False)
                .reset_index().rename(columns={"category": "Category"}))
        score_cols = ["Effectiveness"] + [D.TRAIT_SHORT[k] for k, *_ in D.TRAITS]

        def score_c(v):
            if pd.isna(v):
                return ""
            return cell_style(D.SCALE_5[max(0, min(4, round(v) - 1))])

        def loss_c(v):
            idx = 4 if v < .08 else 3 if v < .14 else 2 if v < .20 else 1 if v < .28 else 0
            return cell_style(D.SCALE_5[idx])
        st.dataframe(
            summ.style.map(score_c, subset=score_cols).map(loss_c, subset=["Loss"])
                .format({**{c: "{:.2f}" for c in score_cols}, "Loss": "{:.0%}"}, na_rep="–"),
            hide_index=True, width="stretch", height=35 * (len(summ) + 1) + 3)

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(key="card_mal"):
        st.subheader("Most common malfunctions")
        m = (df["malfunctions"].fillna("").str.split("; ").explode()
             .loc[lambda x: (x != D.MAL_NONE) & (x != "")]
             .value_counts().rename_axis("type").reset_index(name="reports"))
        if m.empty:
            st.caption("No malfunctions reported.")
        else:
            bars = alt.Chart(m).mark_bar(cornerRadiusEnd=4, color=D.RED).encode(
                x=alt.X("reports:Q", title="Reports"),
                y=alt.Y("type:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
                tooltip=["type", "reports"])
            labels = bars.mark_text(align="left", dx=4, color="#d3dae2").encode(text="reports:Q")
            st.altair_chart((bars + labels).properties(height=alt.Step(28)), width="stretch")
    with c2, st.container(key="card_drivers"):
        st.subheader("What limited the result")
        st.caption("Reports that were not a full success")
        d = (df.loc[df["outcome"] != "Success", "outcome_driver"].fillna("Not given")
             .value_counts().rename_axis("factor").reset_index(name="reports"))
        if d.empty:
            st.caption("Every report was a full success.")
        else:
            bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, color=D.ORANGE).encode(
                x=alt.X("reports:Q", title="Reports"),
                y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                tooltip=["factor", "reports"])
            labels = bars.mark_text(align="left", dx=4, color="#d3dae2").encode(text="reports:Q")
            st.altair_chart((bars + labels).properties(height=alt.Step(28)), width="stretch")

    with st.container(key="card_outcomes"):
        st.subheader("Mission outcomes")
        st.caption(f"Share of reports by {dim_label}")
        o = (df.assign(cond=df[col].fillna("").replace("", NOT_RECORDED))
             .groupby(["cond", "outcome"]).size().reset_index(name="n"))
        found = set(o["cond"])
        y_order = [v for v in order if v in found] + sorted(found - set(order) - {NOT_RECORDED})
        if NOT_RECORDED in found:
            y_order.append(NOT_RECORDED)
        o_order = D.values(D.OUTCOMES)
        chart = alt.Chart(o).mark_bar().encode(
            y=alt.Y("cond:N", sort=y_order, title=None, axis=alt.Axis(labelLimit=200)),
            x=alt.X("n:Q", stack="normalize", title="Share of reports", axis=alt.Axis(format="%")),
            color=alt.Color("outcome:N", sort=o_order,
                            scale=alt.Scale(domain=o_order,
                                            range=[D.color_of(D.OUTCOMES, v) for v in o_order]),
                            legend=alt.Legend(title=None, orient="bottom")),
            order=alt.Order("sort_key:Q"),
            tooltip=[alt.Tooltip("cond:N", title=dim_label), "outcome", "n"],
        ).transform_calculate(sort_key=f"indexof({o_order!r}, datum.outcome)")
        st.altair_chart(chart.properties(height=alt.Step(30)), width="stretch")


# ===========================================================================
# START PAGE: two doors (field report / analyst login)
# ===========================================================================
FLOW_HTML = """
<div style='display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:6px 0 12px;font-size:.92rem'>
 <span class='badge' style='background:rgba(74,155,209,.15);color:#9ccbec;border:1px solid #2c6a95'>Your unit</span>→
 <span class='badge' style='background:rgba(74,155,209,.15);color:#9ccbec;border:1px solid #2c6a95'>Secure upload</span>→
 <span class='badge' style='background:rgba(74,155,209,.15);color:#9ccbec;border:1px solid #2c6a95'>Central database</span>→
 <span class='badge' style='background:rgba(176,127,214,.18);color:#dcc6f2;border:1px solid #6d4a9a'>Analyst cell</span>
</div>
<div class='q-hint'>Field units only file reports. They never see the database or the analysis.</div>"""


def flow_html():
    out = FLOW_HTML
    for txt in ("Your unit", "Secure upload", "Central database", "Analyst cell",
                "Field units only file reports. They never see the database or the analysis."):
        out = out.replace(f">{txt}<", f">{html.escape(t(txt))}<")
    return out


@st.cache_resource
def access_state():
    """Shared across every visitor: how the Analyst login button behaves. Resets on app restart."""
    return {"mode": "after_report"}       # after_report | open | locked


ACCESS_LABELS = {"after_report": "Unlock after filing a report (default)",
                 "open": "Open to everyone", "locked": "Locked for everyone (organizer only)"}


def analyst_allowed():
    if ADMIN_CODE and st.session_state.get("admin_ok"):
        return True
    if not ADMIN_CODE:
        return True                       # local copy with no passcode set: no gate
    mode = access_state()["mode"]
    return mode == "open" or (mode == "after_report" and st.session_state.get("reported", False))


def page_home():
    st.session_state["view"] = None
    add_css(GLOBAL_DOOR_CSS)
    page_header("Battlefield learning platform", "Field reporting prototype. Choose how you're using it.")
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(key="card_door_field"):
        st.markdown("<div class='door-tag' style='color:#9ccbec'>FIELD UNIT</div>", unsafe_allow_html=True)
        st.markdown("### File field report (soldier)")
        st.caption("The short daily form a crew or outpost fills in after an engagement. About 2 minutes.")
        if st.button("File field report", type="primary", icon=":material/edit_note:", key="door_field",
                     width="stretch"):
            st.session_state["view"] = "field"
            st.switch_page(PAGE_NEW)
    allowed = analyst_allowed()
    with c2, st.container(key="card_door_analyst"):
        st.markdown("<div class='door-tag' style='color:#dcc6f2'>AUTHORIZED PERSONNEL</div>",
                    unsafe_allow_html=True)
        st.markdown("### Analyst login")
        st.caption("Report database, insights, scorecard, weapon profiles, sector map and graphs.")
        if st.button("Analyst login" if allowed else "Analyst login (locked)",
                     icon=":material/lock_open:" if allowed else ":material/lock:", key="door_analyst",
                     disabled=not allowed, width="stretch"):
            st.session_state["view"] = "analyst"
            st.switch_page(PAGE_DB)
        if not allowed:
            mode = access_state()["mode"]
            st.caption(":material/info: File a field report first." if mode == "after_report"
                       else ":material/info: Locked by the organizer.")
    st.write("")
    with st.expander("Organizer", icon=":material/admin_panel_settings:"):
        if not ADMIN_CODE:
            st.caption("No organizer passcode is set (BLP_ADMIN_CODE), so Analyst login is always open here.")
        elif not st.session_state.get("admin_ok"):
            admin_unlock()
        else:
            st.radio("Analyst login for everyone", list(ACCESS_LABELS), format_func=ACCESS_LABELS.get,
                     key="access_mode_pick", index=list(ACCESS_LABELS).index(access_state()["mode"]),
                     on_change=lambda: access_state().update(mode=st.session_state["access_mode_pick"]))
            st.caption("You always have analyst access on this device. The setting applies to every "
                       "visitor right away and resets to the default if the app restarts.")


GLOBAL_DOOR_CSS = """
[class*="st-key-card_door_"] { min-height: 230px; }
.door-tag { font-family:'JetBrains Mono', Consolas, monospace; font-size:.78rem; letter-spacing:.12em; font-weight:700; }
.st-key-door_analyst button:disabled { opacity:.45; }
.st-key-door_field button, .st-key-door_analyst button { min-height: 56px; font-size: 1.05rem; }
"""


# ===========================================================================
# APP SHELL
# ===========================================================================
PAGE_HOME = st.Page(page_home, title="Start", icon=":material/home:", default=True)
PAGE_NEW = st.Page(page_new_report, title="New report", icon=":material/edit_note:", url_path="report")
PAGE_DB = st.Page(page_database, title="Report database", icon=":material/table_rows:",
                  url_path="database")
PAGE_INS = st.Page(page_insights, title="Insights", icon=":material/insights:",
                   url_path="insights")


def page_graphs():
    page_header("Graphs", "Interactive views that grow with the reports: networks, learning curves, 3D, "
                "flows and animated bubbles. Pick one below.")
    add_css(".block-container { max-width: 1500px; }")
    sample_tools("graphs")
    if not graphs.render(conn()):
        empty_state()


PAGE_GRAPHS = st.Page(page_graphs, title="Graphs", icon=":material/scatter_plot:", url_path="graphs")

begin_page()
st.session_state["_tr_on"] = False          # Ukrainian only ever switches on inside the report form
# Field view: no tabs at all (start page and form only). Analyst view: every tab, once allowed.
in_analyst = st.session_state.get("view") == "analyst" and analyst_allowed()
pages = [PAGE_HOME, PAGE_NEW] + ([PAGE_DB, PAGE_INS, PAGE_GRAPHS] if analyst_allowed() else [])
nav = st.navigation(pages, position="top" if in_analyst else "hidden")
if st.button("Share", icon=":material/qr_code_2:", key="share_btn", help="Show a QR code to this app"):
    share_dialog()
add_css(".st-key-lang_corner { position: fixed; top: 10px; right: 160px; z-index: 999990; width: auto; }"
        ".st-key-share_btn { position: fixed; top: 10px; right: 64px; z-index: 999990; }"
        ".st-key-share_btn button { min-height: 36px; }")
nav.run()
flush_css()
