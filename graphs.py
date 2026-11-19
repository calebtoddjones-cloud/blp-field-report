"""Graphs page: interactive, visually distinct views of the level 1 reports.

Every view is computed from the live database each time the page opens, so the pictures grow
and change as reports come in. Plain counts and averages: descriptive only, not findings,
and nothing here feeds the scorecard or any model.
"""

import html
import math

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D

BG = "rgba(0,0,0,0)"
FG = "#d3dae2"
MUTED = "#8693a1"
GRID = "#2c3540"
EFF_SCALE = [[0.0, D.RED], [0.25, D.ORANGE], [0.5, D.YELLOW], [0.75, D.LIME], [1.0, D.GREEN]]
CAT_COLORS = ["#4a9bd1", "#e3b93c", "#46b673", "#ec8a3c", "#b07fd6", "#5fc7c2", "#e05d5d",
              "#a3c94a", "#d987b6", "#8d99a6", "#7fa7e0", "#c9a36b", "#6fbf9a", "#ffffff"]


def cat_color_map(cats):
    return {c: CAT_COLORS[i % len(CAT_COLORS)] for i, c in enumerate(sorted(cats))}


def style(fig, height=620, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=FG, size=13),
        margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
        legend=dict(bgcolor="rgba(15,19,24,.6)", bordercolor=GRID, borderwidth=1),
        hoverlabel=dict(bgcolor="#171c23", bordercolor=GRID, font=dict(color=FG)))
    return fig


def note(text):
    st.markdown(f"<div class='q-hint' style='margin:-4px 0 10px'>{html.escape(text)}</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 1. Network
# ---------------------------------------------------------------------------
def network_view(cards, df):
    mode = st.segmented_control(
        "Link by", ["Used together", "Platform → malfunction", "Category hubs"],
        default="Used together", key="g_net_mode")
    min_w = st.slider("Hide links seen fewer than this many times", 1, 5, 1, key="g_net_min")
    note({
        "Used together": "Each dot is a platform; a line joins two platforms reported in the same "
                         "engagement. Thicker = more often together. Size = how often it was reported, "
                         "colour = its average effectiveness. Clusters show platforms that get fielded as a package.",
        "Platform → malfunction": "Red hubs are malfunction types; each platform links to the problems "
                                   "reported alongside it. Big red hubs with many spokes are the problems "
                                   "that cut across many systems.",
        "Category hubs": "Each category is a hub; its platforms sit around it, linked to the conditions "
                         "(enemy EW level) they were used under. Shows which families carry most of the load "
                         "under heavy jamming.",
    }[mode or "Used together"])
    G = nx.Graph()
    stats = cards.groupby("platform").agg(n=("p_effectiveness", "size"), eff=("p_effectiveness", "mean"),
                                          category=("category", "first"))
    if mode == "Platform → malfunction":
        for p, r in stats.iterrows():
            G.add_node(p, kind="platform", n=r.n, eff=r.eff, category=r.category)
        for _, row in cards.iterrows():
            for m in str(row["malfunctions"] or "").split("; "):
                if m and m != D.MAL_NONE:
                    G.add_node("⚠ " + m, kind="mal")
                    w = G.get_edge_data(row["platform"], "⚠ " + m, {"w": 0})["w"]
                    G.add_edge(row["platform"], "⚠ " + m, w=w + 1)
    elif mode == "Category hubs":
        for p, r in stats.iterrows():
            G.add_node(p, kind="platform", n=r.n, eff=r.eff, category=r.category)
            G.add_node("■ " + r.category, kind="hub", category=r.category)
            G.add_edge("■ " + r.category, p, w=int(r.n))
        for _, row in cards.iterrows():
            ew = row["ew_level"]
            if ew:
                G.add_node("EW " + ew, kind="cond")
                w = G.get_edge_data(row["platform"], "EW " + ew, {"w": 0})["w"]
                G.add_edge(row["platform"], "EW " + ew, w=w + 1)
    else:
        for p, r in stats.iterrows():
            G.add_node(p, kind="platform", n=r.n, eff=r.eff, category=r.category)
        for _, grp in cards.groupby("id"):
            ps = sorted(set(grp["platform"]))
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    w = G.get_edge_data(ps[i], ps[j], {"w": 0})["w"]
                    G.add_edge(ps[i], ps[j], w=w + 1)
    G.remove_edges_from([(a, b) for a, b, d in G.edges(data=True) if d["w"] < min_w])
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    if G.number_of_nodes() == 0:
        st.caption("Not enough links yet. Lower the threshold or add reports.")
        return
    spread = 3.2 if mode == "Platform → malfunction" else 1.6
    pos = nx.spring_layout(G, seed=7, k=spread / math.sqrt(G.number_of_nodes()), iterations=150, weight=None)
    fig = go.Figure()
    # edges, bucketed by weight so thickness varies without one trace per edge
    edges = list(G.edges(data=True))
    wmax = max(d["w"] for *_, d in edges) if edges else 1
    for lo, hi, width, alpha in [(0, .34, .6, .25), (.34, .67, 1.4, .4), (.67, 1.01, 2.6, .6)]:
        xs, ys = [], []
        for a, b, d in edges:
            f = d["w"] / wmax
            if lo <= f < hi:
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(width=width, color=f"rgba(200,210,222,{alpha})")))
    plats = [n for n, d in G.nodes(data=True) if d.get("kind") == "platform"]
    if plats:
        sizes = [8 + 22 * math.sqrt(G.nodes[n]["n"] / stats["n"].max()) for n in plats]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in plats], y=[pos[n][1] for n in plats], mode="markers",
            name="Platforms", text=plats,
            customdata=[[G.nodes[n]["category"], G.nodes[n]["n"], G.nodes[n]["eff"], G.degree(n)] for n in plats],
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]} ratings · avg "
                          "%{customdata[2]:.2f}<br>%{customdata[3]} links<extra></extra>",
            marker=dict(size=sizes, color=[G.nodes[n]["eff"] for n in plats], colorscale=EFF_SCALE,
                        cmin=1, cmax=5, line=dict(width=1, color="#0f1318"),
                        colorbar=dict(title="Avg eff.", thickness=12, len=.6))))
    for kind, color, label, size in [("mal", D.RED, "Malfunctions", 26), ("hub", "#ffffff", "Categories", 24),
                                     ("cond", D.BLUE, "Enemy EW", 24)]:
        ns = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        if not ns:
            continue
        cmap = cat_color_map(stats["category"].unique()) if kind == "hub" else None
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", name=label,
            text=ns, textposition="top center", textfont=dict(size=11, color=FG),
            customdata=[G.degree(n, weight="w") for n in ns],
            hovertemplate="<b>%{text}</b><br>%{customdata} links<extra></extra>",
            marker=dict(size=size, symbol="diamond" if kind != "mal" else "circle",
                        color=[cmap[G.nodes[n]["category"]] for n in ns] if cmap else color,
                        line=dict(width=2, color="#0f1318"))))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    st.plotly_chart(style(fig, 680), width="stretch", key="g_net_fig")
    st.caption(f"{len(plats)} platforms · {G.number_of_edges()} links. Drag to pan, scroll to zoom, "
               "hover a dot for its numbers.")


# ---------------------------------------------------------------------------
# 2. Learning curve over time
# ---------------------------------------------------------------------------
def timeline_view(cards, df):
    view = st.segmented_control("Show", ["Running average", "Reports piling up"],
                                default="Running average", key="g_time_view")
    c = cards.sort_values(["report_date", "id"]).copy()
    cats = c["category"].value_counts()
    pick = st.multiselect("Categories", list(cats.index), default=list(cats.index[:6]), key="g_time_cats")
    c = c[c["category"].isin(pick)]
    cmap = cat_color_map(cards["category"].unique())
    fig = go.Figure()
    if view == "Reports piling up":
        note("Cumulative platform ratings by category as reports arrive. Steeper = more reporting "
             "on that family. A flat line means the platform family has gone quiet.")
        for cat in pick:
            s = c[c["category"] == cat].groupby("report_date").size().cumsum()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=cat, stackgroup="one",
                                     line=dict(width=1.2, color=cmap[cat])))
        fig.update_yaxes(title="Ratings so far", gridcolor=GRID)
    else:
        note("Each line is a category's average effectiveness using only the reports received up to that "
             "date. Early on the lines swing wildly; as reports pile up they settle. That settling is the "
             "platform 'learning'. The shaded band is roughly where the true average could still be.")
        for cat in pick:
            s = c[c["category"] == cat]
            if s.empty:
                continue
            vals = s["p_effectiveness"].astype(float).to_numpy()
            n = np.arange(1, len(vals) + 1)
            mean = np.cumsum(vals) / n
            sq = np.cumsum(vals ** 2) / n
            sd = np.sqrt(np.maximum(sq - mean ** 2, 0.25))
            half = 1.64 * sd / np.sqrt(n)
            x = s["report_date"].to_numpy()
            col = cmap[cat]
            r, g, b = (int(col[i:i + 2], 16) for i in (1, 3, 5))
            fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                     y=np.concatenate([mean + half, (mean - half)[::-1]]),
                                     fill="toself", fillcolor=f"rgba({r},{g},{b},.10)", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=cat, line=dict(width=2.4, color=col),
                                     customdata=n, hovertemplate=f"<b>{html.escape(cat)}</b><br>%{{x}}<br>"
                                     "avg %{y:.2f} after %{customdata} ratings<extra></extra>"))
        fig.update_yaxes(title="Average effectiveness so far", range=[1, 5.2], gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    st.plotly_chart(style(fig, 560), width="stretch", key="g_time_fig")


# ---------------------------------------------------------------------------
# 3. 3D views
# ---------------------------------------------------------------------------
TRAIT_COLS = {"Effectiveness": "p_effectiveness",
              **{D.TRAIT_SHORT[k]: f"p_{k}" for k, *_ in D.TRAITS}}


def three_d_view(cards, df):
    view = st.segmented_control("3D view", ["Effectiveness landscape", "Trait space"],
                                default="Effectiveness landscape", key="g_3d_view")
    if view == "Trait space":
        note("Each ball is a platform placed by three of its average traits; size = how many ratings, "
             "colour = category. Drag to rotate. Balls far from the rest are the outliers worth a look; "
             "a tight clump means a family of systems that behave alike.")
        c1, c2, c3, c4 = st.columns(4)
        names = list(TRAIT_COLS)
        x = c1.selectbox("X axis", names, index=names.index("Lethality"), key="g_3d_x")
        y = c2.selectbox("Y axis", names, index=names.index("Platform survival"), key="g_3d_y")
        z = c3.selectbox("Z axis", names, index=0, key="g_3d_z")
        min_n = c4.slider("Min ratings", 1, 10, 2, key="g_3d_min")
        agg = cards.groupby(["platform", "category"]).agg(
            n=("p_effectiveness", "size"), **{k: (v, "mean") for k, v in TRAIT_COLS.items()}).reset_index()
        agg = agg[agg["n"] >= min_n].dropna(subset=[x, y, z])
        if agg.empty:
            st.caption("No platform has enough ratings yet.")
            return
        cmap = cat_color_map(cards["category"].unique())
        fig = go.Figure()
        for cat, g in agg.groupby("category"):
            fig.add_trace(go.Scatter3d(
                x=g[x], y=g[y], z=g[z], mode="markers", name=cat, text=g["platform"],
                customdata=g["n"],
                hovertemplate=f"<b>%{{text}}</b><br>{x} %{{x:.2f}}<br>{y} %{{y:.2f}}<br>{z} %{{z:.2f}}"
                              "<br>%{customdata} ratings<extra></extra>",
                marker=dict(size=np.clip(7 + 3 * np.sqrt(g["n"]), 7, 26), color=cmap[cat], opacity=.9,
                            line=dict(width=.5, color="#0f1318"))))
        axis = dict(range=[1, 5], gridcolor=GRID, backgroundcolor="rgba(23,28,35,.6)", showbackground=True,
                    color=FG)
        fig.update_layout(scene=dict(xaxis=dict(title=x, **axis), yaxis=dict(title=y, **axis),
                                     zaxis=dict(title=z, **axis), camera=dict(eye=dict(x=1.35, y=1.2, z=.75))),
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(style(fig, 700), width="stretch", key="g_3d_trait")
        return
    note("A terrain of average effectiveness: categories along one side, enemy EW level along the other, "
         "height and colour = average effectiveness. Drag to rotate. Valleys show where a family of systems "
         "breaks down; a ridge that stays high under heavy jamming is a robust family.")
    levels = [v for v in D.values(D.EW_LEVELS)]
    g = cards[cards["ew_level"].isin(levels)].groupby(["category", "ew_level"])["p_effectiveness"].mean().unstack()
    g = g.reindex(columns=levels)
    if g.empty:
        st.caption("No ratings with an enemy EW level yet.")
        return
    drop = (g[levels[0]] - g[levels[-1]]).fillna(0)
    g = g.loc[drop.sort_values().index]                       # robust families at the front
    z = g.to_numpy(dtype=float)
    fig = go.Figure(go.Surface(
        z=z, x=levels, y=list(g.index), colorscale=EFF_SCALE, cmin=1, cmax=5, connectgaps=True,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
        hovertemplate="<b>%{y}</b><br>EW %{x}<br>avg %{z:.2f}<extra></extra>",
        colorbar=dict(title="Avg eff.", thickness=12, len=.6)))
    axis = dict(gridcolor=GRID, backgroundcolor="rgba(23,28,35,.6)", showbackground=True, color=FG)
    fig.update_layout(scene=dict(xaxis=dict(title="Enemy EW", **axis), yaxis=dict(title="", **axis),
                                 zaxis=dict(title="Avg effectiveness", range=[1, 5], **axis),
                                 camera=dict(eye=dict(x=1.7, y=-1.5, z=1.0)), aspectratio=dict(x=1, y=1.6, z=.7)))
    st.plotly_chart(style(fig, 720, legend=False), width="stretch", key="g_3d_surf")


# ---------------------------------------------------------------------------
# 4. Flow (Sankey)
# ---------------------------------------------------------------------------
def flow_view(cards, df):
    start = st.segmented_control("Start the flow from", ["Enemy EW", "Platform category", "Primary task"],
                                 default="Enemy EW", key="g_flow_from")
    note("Follow the reports from left to right: what they faced, how the mission ended, and (for anything "
         "short of success) what most limited it. Band width = number of reports. Hover a band for the count.")
    d = df.copy()
    if start == "Platform category":
        first = cards.drop_duplicates("id").set_index("id")["category"]
        d["a"] = d["id"].map(first).fillna("Unlisted / custom")
    elif start == "Primary task":
        d["a"] = d["task"]
    else:
        d["a"] = "EW " + d["ew_level"].fillna("not given")
    d["b"] = d["outcome"]
    d["c"] = np.where(d["outcome"] == "Success", None, d["outcome_driver"].fillna("Not given"))
    labels, idx = [], {}

    def node(name):
        if name not in idx:
            idx[name] = len(labels)
            labels.append(name)
        return idx[name]
    src, dst, val, col = [], [], [], []
    for (a, b), n in d.groupby(["a", "b"]).size().items():
        src.append(node(a)); dst.append(node(b)); val.append(int(n))
        c = D.color_of(D.OUTCOMES, b)
        col.append(f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},.35)")
    for (b, c), n in d.dropna(subset=["c"]).groupby(["b", "c"]).size().items():
        src.append(node(b)); dst.append(node("▸ " + c)); val.append(int(n)); col.append("rgba(236,138,60,.30)")
    ncol = [D.color_of(D.OUTCOMES, l) if l in D.values(D.OUTCOMES) else
            (D.ORANGE if l.startswith("▸") else D.color_of(D.THREAT, l[3:]) if l.startswith("EW ") else D.BLUE)
            for l in labels]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=ncol, pad=16, thickness=18, line=dict(color="#0f1318", width=1)),
        link=dict(source=src, target=dst, value=val, color=col,
                  hovertemplate="%{source.label} → %{target.label}<br>%{value} reports<extra></extra>")))
    st.plotly_chart(style(fig, 620, legend=False), width="stretch", key="g_flow_fig")


# ---------------------------------------------------------------------------
# 5. Animated bubbles
# ---------------------------------------------------------------------------
def bubbles_view(cards, df):
    note("Press play. Each bubble is a platform category; it moves as reports arrive week by week "
         "(cumulative). Across = malfunction rate, up = average effectiveness, size = number of ratings. "
         "Good place: top left. Watch which families drift toward the bottom right as evidence builds.")
    c = cards.copy()
    c["week"] = pd.to_datetime(c["report_date"]).dt.to_period("W").dt.start_time
    c["mal"] = (c["malfunctions"].fillna("") != D.MAL_NONE).astype(float)
    weeks = sorted(c["week"].unique())
    if len(weeks) < 2:
        st.caption("Needs reports from at least two different weeks.")
        return
    cats = sorted(c["category"].unique())
    cmap = cat_color_map(cats)
    rows = []
    for w in weeks:
        upto = c[c["week"] <= w]
        g = upto.groupby("category").agg(n=("p_effectiveness", "size"), eff=("p_effectiveness", "mean"),
                                         mal=("mal", "mean"))
        for cat, r in g.iterrows():
            rows.append((w, cat, r.n, r.eff, r.mal))
    a = pd.DataFrame(rows, columns=["week", "category", "n", "eff", "mal"])
    nmax = a["n"].max()

    def trace(sub):
        return go.Scatter(
            x=sub["mal"], y=sub["eff"], mode="markers+text", text=sub["category"], textposition="top center",
            textfont=dict(size=10, color=MUTED), customdata=sub["n"],
            hovertemplate="<b>%{text}</b><br>malfunctions %{x:.0%}<br>avg %{y:.2f}<br>%{customdata} ratings<extra></extra>",
            marker=dict(size=10 + 50 * np.sqrt(sub["n"] / nmax), color=[cmap[k] for k in sub["category"]],
                        opacity=.85, line=dict(width=1, color="#0f1318")))
    frames = [go.Frame(data=[trace(a[a["week"] == w])], name=pd.Timestamp(w).strftime("%Y-%m-%d"))
              for w in weeks]
    fig = go.Figure(data=[trace(a[a["week"] == weeks[0]])], frames=frames)
    steps = [dict(method="animate", label=f.name,
                  args=[[f.name], dict(mode="immediate", frame=dict(duration=500, redraw=True),
                                       transition=dict(duration=400))]) for f in frames]
    fig.update_layout(
        xaxis=dict(title="Malfunction rate", tickformat=".0%", range=[-.03, max(.6, a["mal"].max() + .05)],
                   gridcolor=GRID),
        yaxis=dict(title="Average effectiveness", range=[1, 5.2], gridcolor=GRID),
        updatemenus=[dict(type="buttons", x=0, y=-.12, xanchor="left", direction="left",
                          bgcolor="#212831", font=dict(color=FG),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=600, redraw=True),
                                                         transition=dict(duration=400), fromcurrent=True)]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, steps=steps, x=.16, y=-.08, len=.84, currentvalue=dict(prefix="Week of "),
                      font=dict(color=FG), bgcolor="#212831")])
    st.plotly_chart(style(fig, 640, legend=False), width="stretch", key="g_bub_fig")


VIEWS = {
    "Network": ("hub", network_view),
    "Learning curve": ("timeline", timeline_view),
    "3D": ("view_in_ar", three_d_view),
    "Flow": ("account_tree", flow_view),
    "Bubbles over time": ("bubble_chart", bubbles_view),
}


def render(conn):
    df = D.reports_df(conn)
    cards = D.platform_rows_df(conn)
    cards = cards[(cards["rated_individually"] == 1) & cards["p_effectiveness"].notna()]
    if df.empty or cards.empty:
        st.info("No level 1 reports yet. These graphs build themselves from the reports.")
        return False
    st.html("<div class='caveat'><span>⚠</span><span>Built from the live level 1 reports and redrawn every "
            "time: the more reports, the richer the pictures. If sample data is loaded it is <b>synthetic</b>, "
            "so any pattern is planted, not a finding. Plain counts and averages.</span></div>")
    pick = st.segmented_control("Graph", list(VIEWS), default="Network", key="g_pick",
                                format_func=lambda k: f":material/{VIEWS[k][0]}: {k}")
    VIEWS[pick or "Network"][1](cards, df)
    return True
