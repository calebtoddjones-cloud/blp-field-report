"""Lab page: sketch a new system from one the model already knows, set the conditions, see a forecast.

How it works
- A random forest is trained on the level 1 platform cards in the live database: effectiveness (1-5)
  from platform category, origin, field-modified, and the battlefield conditions on the report.
  After-the-fact observations (traits, malfunctions, outcome) are never inputs: a forecast has to work
  before the engagement, and in the sample data the traits are generated from effectiveness.
- Holdout split is by REPORT (seed 42), and the forest is compared with guessing the average.
  The forest you play with is the one that was tested.
- A "new system" starts from a known platform (its category and origin). Modifications the model can
  represent: a different origin, a field-modified build, and "behaves partly like another class",
  which blends the two classes' forecasts. The blend is an ANALYST ASSUMPTION, not something learned.
- The model cannot see range, seeker, speed or any design detail. Say so on the page.

SYNTHETIC sample data has a planted EW pattern; results test the pipeline, they are not findings.
"""

import hashlib
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import data as D
from graphs import FG, GRID, MUTED, style

SEED = 42
MIN_CARDS = 60          # below this the page asks for more reports instead of training
THIN_CLASS = 20         # fewer cards than this in a class -> "thin data" warning
TARGET = "p_effectiveness"
CAT_FEATS = ["category", "origin", "ew_level", "ad_level", "time_of_day", "weather", "terrain",
             "unit_type", "task", "enemy_type", "enemy_strength", "enemy_adaptation"]
NUM_FEATS = ["p_field_modified"]
COND_LABELS = {"ew_level": "Enemy EW / jamming", "ad_level": "Enemy air defense", "time_of_day": "Time of day",
               "weather": "Weather", "terrain": "Terrain", "unit_type": "Unit type", "task": "Primary task",
               "enemy_type": "Enemy type", "enemy_strength": "Enemy strength",
               "enemy_adaptation": "Enemy adaptation"}
LEVELS = [v for v in D.values(D.THREAT) if v != "Unknown"]          # None, Low, Medium, High
ORIGINS = ["U.S.", "Ukraine", "Other partner"]
NEW_COLOR, REF_COLOR, ALT_COLOR = "#b07fd6", "#4a9bd1", "#e3b93c"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _prep(df):
    X = df.reindex(columns=CAT_FEATS + NUM_FEATS).copy()
    for c in CAT_FEATS:
        X[c] = X[c].fillna("Unknown").astype(str).replace({"": "Unknown", "None": "None"})
    for c in NUM_FEATS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X


@st.cache_resource(show_spinner="Training the model on the current reports…", max_entries=4)
def _train(_key, _cards):
    X, y, groups = _prep(_cards), _cards[TARGET].astype(float), _cards["id"]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
                             ("num", "passthrough", NUM_FEATS)])
    forest = Pipeline([("pre", pre), ("model", RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED))]).fit(X.iloc[tr], y.iloc[tr])
    base = DummyRegressor(strategy="mean").fit(X.iloc[tr], y.iloc[tr])
    mae_base = mean_absolute_error(y.iloc[te], base.predict(X.iloc[te]))
    mae_forest = mean_absolute_error(y.iloc[te], forest.predict(X.iloc[te]))
    return {"pipe": forest, "mae_base": float(mae_base), "mae_forest": float(mae_forest),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "reports_test": int(groups.iloc[te].nunique()), "reports": int(groups.nunique())}


def _trees(model, rows):
    """Per-tree predictions, shape (n_trees, n_rows)."""
    pipe = model["pipe"]
    Xt = pipe["pre"].transform(_prep(pd.DataFrame(rows)))
    return np.stack([t.predict(Xt) for t in pipe["model"].estimators_])


def _forecast(model, rows, alt_rows=None, w=0.0):
    """Mean and 10-90% tree spread per row; optionally blended with a second class's rows (weight w)."""
    t = _trees(model, rows)
    if alt_rows is not None and w > 0:
        t = (1 - w) * t + w * _trees(model, alt_rows)
    return t.mean(axis=0), np.percentile(t, 10, axis=0), np.percentile(t, 90, axis=0)


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------
def _eff_color(v):
    return D.SCALE_5[int(min(4, max(0, round(v) - 1)))]


def _card(label, value, sub, color, accent):
    return (f"<div style='border:1px solid {GRID};border-left:4px solid {accent};border-radius:10px;"
            f"padding:12px 16px;background:rgba(23,28,35,.6)'>"
            f"<div style='font-size:.78rem;letter-spacing:.08em;color:{MUTED};font-weight:700'>{html.escape(label)}</div>"
            f"<div style='font-size:2.1rem;font-weight:800;color:{color};line-height:1.15'>{value}</div>"
            f"<div style='font-size:.85rem;color:{MUTED}'>{sub}</div></div>")


def _note(text):
    st.markdown(f"<div class='q-hint' style='margin:-4px 0 10px'>{html.escape(text)}</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render(conn):
    cards = D.platform_rows_df(conn)
    cards = cards[(cards["rated_individually"] == 1) & cards["p_effectiveness"].notna()].reset_index(drop=True)
    if len(cards) < MIN_CARDS:
        st.info(f"The Lab trains a model on the reports, so it needs at least {MIN_CARDS} rated platform cards "
                f"(there are {len(cards)}). Add reports, or generate sample data above.", icon=":material/science:")
        return False

    key = hashlib.md5(cards[["id", TARGET] + CAT_FEATS + NUM_FEATS].to_csv(index=False).encode()).hexdigest()
    model = _train(key, cards)
    synthetic = bool((cards["notes"].fillna("") == "[sample data]").all())

    st.html("<div class='caveat'><span>⚠</span><span>A forecast from the reports, for exploring ideas. "
            + ("Everything here is <b>synthetic sample data with a planted EW pattern</b>: this shows the method "
               "working, it says nothing about real weapons. " if synthetic else
               "Some or all of these reports are synthetic sample data; treat the numbers as a method demo. ")
            + "The model only knows a system's class, origin, whether it was field-modified, and the "
              "conditions below.</span></div>")

    counts = cards["platform"].value_counts()
    known = sorted(counts.index, key=lambda n: (D.PLATFORM_INFO.get(n, ("~",))[0], n))
    default = "HAWK" if "HAWK" in counts else counts.index[0]
    left, right = st.columns([5, 7], gap="large")

    # ------------------------------------------------------------- controls
    with left:
        st.markdown("#### 1 · Start from a system the model knows")
        base = st.selectbox("Closest existing system", known, index=known.index(default), key="lab_base",
                            format_func=lambda n: f"{n}  ·  {counts[n]} report{'s' if counts[n] != 1 else ''}")
        row0 = cards[cards["platform"] == base].iloc[0]
        base_cat, base_origin = row0["category"], row0["origin"]
        n_cat = int((cards["category"] == base_cat).sum())
        st.caption(f"Class: **{base_cat}** ({n_cat} reports in this class) · Origin: **{base_origin}**")
        name = st.text_input("Name the new system", value=f"{base} (modified)", key=f"lab_name_{base}")

        st.markdown("#### 2 · Modifications")
        origin_opts = [base_origin] + [o for o in ORIGINS if o != base_origin]
        origin = st.segmented_control("Built / supplied by", origin_opts, default=base_origin, key=f"lab_origin_{base}")
        field_mod = st.toggle("Field-modified build", key="lab_fieldmod",
                              help="Same question as the report form: was the system modified in the field?")
        other_classes = ["None"] + sorted(c for c in cards["category"].unique() if c != base_cat)
        alt_cat = st.selectbox("Also behaves like another class", other_classes, key=f"lab_alt_{base}",
                               help="Example: a HAWK with a drone-style data link might behave partly like "
                                    "Counter-UAS or a UAS class.")
        w = 0.0
        if alt_cat != "None":
            w = st.slider(f"How much like {alt_cat}", 0, 100, 40, step=10, format="%d%%", key="lab_w") / 100
            _note("Analyst assumption: the forecast blends the two classes' learned behaviour in this ratio. "
                  "The model has not seen this hybrid.")

        st.markdown("#### 3 · Battlefield conditions")
        cond = {
            "ew_level": st.select_slider("Enemy EW / jamming", LEVELS, value="None", key="lab_ew"),
            "ad_level": st.select_slider("Enemy air defense", LEVELS, value="Low", key="lab_ad"),
            "time_of_day": st.segmented_control("Time of day", D.values(D.TIME_OF_DAY), default="Day", key="lab_tod")
                           or "Day",
            "weather": st.select_slider("Weather", D.values(D.WEATHER), value="Clear", key="lab_wx"),
            "terrain": st.selectbox("Terrain", D.values(D.TERRAIN), key="lab_terrain"),
        }
        with st.expander("More conditions (unit, task, enemy)"):
            cond["unit_type"] = st.selectbox("Unit type", D.UNIT_TYPES, key="lab_unit")
            cond["task"] = st.selectbox("Primary task", D.TASKS, index=D.TASKS.index("Defense"), key="lab_task")
            cond["enemy_type"] = st.selectbox("Enemy type", D.ENEMY_TYPES, key="lab_etype")
            cond["enemy_strength"] = st.select_slider("Enemy strength", D.values(D.ENEMY_STRENGTH), key="lab_estr")
            cond["enemy_adaptation"] = st.selectbox("Enemy adaptation", D.values(D.ENEMY_ADAPT), key="lab_eadapt")

    # ------------------------------------------------------------- forecast
    new_row = {"category": base_cat, "origin": origin or base_origin, "p_field_modified": int(field_mod), **cond}
    ref_row = {"category": base_cat, "origin": base_origin, "p_field_modified": 0, **cond}
    alt_row = {**new_row, "category": alt_cat} if alt_cat != "None" else None
    blend = (lambda rows: [{**r, "category": alt_cat} for r in rows]) if alt_row else (lambda rows: None)

    m_new, lo_new, hi_new = _forecast(model, [new_row], blend([new_row]), w)
    m_ref, lo_ref, hi_ref = _forecast(model, [ref_row])
    m_new, lo_new, hi_new, m_ref = float(m_new[0]), float(lo_new[0]), float(hi_new[0]), float(m_ref[0])
    delta = round(m_new - m_ref, 1) + 0.0     # + 0.0 turns -0.0 into 0.0

    with right:
        c1, c2 = st.columns(2)
        c1.html(_card(name.upper()[:40] or "NEW SYSTEM", f"{m_new:.1f}<span style='font-size:1rem;color:{MUTED}'> / 5</span>",
                      f"predicted effectiveness · 80% of trees {lo_new:.1f}–{hi_new:.1f}", _eff_color(m_new), NEW_COLOR))
        c2.html(_card(f"{base.upper()} AS FIELDED", f"{m_ref:.1f}<span style='font-size:1rem;color:{MUTED}'> / 5</span>",
                      f"same conditions · change {delta:+.1f}", _eff_color(m_ref), REF_COLOR))

        # Stress test across enemy EW
        st.markdown("##### Stress test: how it holds up as enemy jamming rises")
        sweep = lambda r: [{**r, "ew_level": lv} for lv in LEVELS]
        s_new = _forecast(model, sweep(new_row), blend(sweep(new_row)), w)
        s_ref = _forecast(model, sweep(ref_row))
        fig = go.Figure()
        series = [(name or "New system", s_new, NEW_COLOR), (f"{base} as fielded", s_ref, REF_COLOR)]
        if alt_row:
            series.append((f"Typical {alt_cat}", _forecast(model, sweep({**ref_row, "category": alt_cat})), ALT_COLOR))
        for label, (m, lo, hi), col in series:
            fig.add_trace(go.Scatter(x=LEVELS + LEVELS[::-1], y=list(hi) + list(lo[::-1]), fill="toself",
                                     fillcolor=col, opacity=.13, line=dict(width=0), hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=LEVELS, y=m, name=label, mode="lines+markers",
                                     line=dict(color=col, width=3), marker=dict(size=9),
                                     hovertemplate="%{x}: %{y:.2f}<extra>" + html.escape(label) + "</extra>"))
        fig.add_vline(x=LEVELS.index(cond["ew_level"]), line=dict(color=MUTED, dash="dot"))
        fig.update_yaxes(range=[1, 5], title="Predicted effectiveness", gridcolor=GRID, dtick=1)
        fig.update_xaxes(title="Enemy EW / jamming", gridcolor=GRID)
        st.plotly_chart(style(fig, height=330), width="stretch", config={"displayModeBar": False})
        _note("Lines: the forest's average forecast at each jamming level, other settings as chosen. "
              "Shaded: where 80% of its 200 trees land. Dotted line: the level you picked.")

        # What moves this forecast
        st.markdown("##### What moves this forecast")
        rows = []
        for f, label in COND_LABELS.items():
            opts = LEVELS if f in ("ew_level", "ad_level") else (
                D.values(D.FIELD_SPECS[f]) if f in D.FIELD_SPECS else
                {"unit_type": D.UNIT_TYPES, "task": D.TASKS, "enemy_type": D.ENEMY_TYPES}[f])
            var = [{**new_row, f: o} for o in opts]
            m, _, _ = _forecast(model, var, blend(var), w)
            rows.append((label, float(m.min()), float(m.max()), opts[int(m.argmin())], opts[int(m.argmax())]))
        rows.sort(key=lambda r: r[2] - r[1])
        fig2 = go.Figure()
        for label, lo, hi, worst, best in rows:
            fig2.add_trace(go.Scatter(
                x=[lo, hi], y=[label, label], mode="lines+markers", showlegend=False,
                line=dict(color=NEW_COLOR, width=8), marker=dict(size=10, color=[D.RED, D.GREEN]),
                hovertemplate=f"{html.escape(label)}<br>worst {html.escape(str(worst))}: {lo:.2f}"
                              f"<br>best {html.escape(str(best))}: {hi:.2f}<extra></extra>"))
        fig2.add_vline(x=m_new, line=dict(color=MUTED, dash="dot"))
        fig2.update_xaxes(range=[1, 5], title="Predicted effectiveness: worst to best option", gridcolor=GRID)
        fig2.update_yaxes(gridcolor=GRID)
        st.plotly_chart(style(fig2, height=340, legend=False), width="stretch", config={"displayModeBar": False})
        _note("Each bar: change one condition through all its options, keeping the others as set. "
              "Long bars are the conditions this system is sensitive to (red = worst option, green = best).")

        # Evidence
        gain = 1 - model["mae_forest"] / model["mae_base"] if model["mae_base"] else 0
        thin = n_cat < THIN_CLASS or (alt_row and int((cards["category"] == alt_cat).sum()) < THIN_CLASS)
        st.markdown("##### Can you trust it?")
        st.markdown(
            f"- Trained on **{model['n_train']}** platform cards; tested on **{model['n_test']}** cards from "
            f"**{model['reports_test']}** reports it never saw. Average miss **{model['mae_forest']:.2f}** points vs "
            f"**{model['mae_base']:.2f}** for always guessing the average (**{gain:.0%} {'smaller' if gain >= 0 else 'larger'}**).\n"
            f"- **{n_cat}** reports for {base_cat}"
            + (f", **{int((cards['category'] == alt_cat).sum())}** for {alt_cat}" if alt_row else "")
            + (". :orange[Thin data: treat this class's forecast as a rough sketch.]" if thin else ".") + "\n"
            "- The model cannot see range, seeker, speed or any design detail. A modification that changes "
            "those only shows up once units report on the new system (or a new question captures it).")
        if gain <= 0:
            st.warning("On this data the forest does no better than guessing the average, so the forecasts "
                       "above are not informative yet. More reports are needed.", icon=":material/warning:")
    return True
