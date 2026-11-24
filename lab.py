"""Lab page: design a new system from one the models already know, set the conditions, see forecasts.

How it works
- Every platform has an estimated capability profile (profiles.py: protection, signature, mobility,
  guidance / control link, jamming resistance, reach, crew). ESTIMATES, illustrative, not authoritative.
  The profile is looked up by platform name at analysis time; the report form is unchanged.
- Models are trained on the level 1 platform cards in the live database. Inputs: platform class, origin,
  field-modified, the capability profile, and the battlefield conditions on the report. Outputs, one model
  each: effectiveness, platform survival, crew survival, ease of maintenance, how hard to hit (all 1-5 ratings
  the form already collects) and loss risk (platform status "Lost / destroyed"). Ratings are outcomes here,
  never inputs.
- Effectiveness gets three models side by side: decision tree, random forest (200 trees, the main forecast)
  and ridge regression with class x condition and link x EW interactions (alpha fixed, not tuned).
- Holdout split is by REPORT (seed 42); every model is compared with a trivial baseline (the average, or the
  base rate for loss). The models you play with are the ones that were tested.
- A new design = the base platform's profile with any characteristic changed, plus origin, field-modified and
  "behaves partly like another class" (blends the two classes' forecasts: an ANALYST ASSUMPTION).
- Guardrails: nearest real systems by profile, an extrapolation alert when no reported system is close, and
  (on synthetic data) a planted-pattern check that the models recover the effects the generator planted.
- 3D: the base platform's model (and a wireframe of the blended class) from the slimmed models already
  embedded in analyst_base.html, drawn with three.js from a CDN. Modifications are not drawn.

SYNTHETIC sample data has planted patterns; results test the pipeline, they are not findings.
"""

import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

import data as D
import profiles as P
from graphs import FG, style

SEED = 42
MIN_CARDS = 60          # below this the page asks for more reports instead of training
THIN_CLASS = 20         # fewer cards than this in a class -> "thin data" warning
AGREE = 0.35            # three models within this many points of each other -> "agree"
FAR = 0.22              # mean profile distance to the nearest reported system above this -> extrapolation
RIDGE_ALPHA = 10.0      # fixed, not tuned
TARGET = "p_effectiveness"
CONDS = ["ew_level", "ad_level", "time_of_day", "weather", "terrain", "unit_type", "task", "enemy_type",
         "enemy_strength", "enemy_adaptation"]
CAT_FEATS = ["category", "origin"] + P.CATEGORICAL + CONDS
NUM_FEATS = ["p_field_modified"] + P.ORDINAL
PAIRS = [("category", c) for c in CONDS] + [("link", "ew_level"), ("mobility", "ad_level")]
LIN_CATS = CAT_FEATS + [f"{a} x {b}" for a, b in PAIRS]
OUTCOMES = {  # target column: (label, short, higher is better)
    "p_effectiveness": ("Effectiveness", "EFF", True),
    "p_rating_survivability": ("Platform survival", "SURV", True),
    "p_rating_crew_survivability": ("Crew survival", "CREW", True),
    "p_rating_ease": ("Ease of maintenance", "MAINT", True),
    "p_rating_hard_to_hit": ("Hard to hit", "EVADE", True),
    "lost": ("Loss risk", "LOSS", False),
}
COND_LABELS = {"ew_level": "Enemy EW / jamming", "ad_level": "Enemy air defense", "time_of_day": "Time of day",
               "weather": "Weather", "terrain": "Terrain", "unit_type": "Unit type", "task": "Primary task",
               "enemy_type": "Enemy type", "enemy_strength": "Enemy strength",
               "enemy_adaptation": "Enemy adaptation"}
LEVELS = [v for v in D.values(D.THREAT) if v != "Unknown"]          # None, Low, Medium, High
ORIGINS = ["U.S.", "Ukraine", "Other partner"]
MODEL_NAMES = ["Decision tree", "Random forest", "Ridge + interactions"]
MODEL_BLURB = {"Decision tree": "one set of yes/no questions",
               "Random forest": "200 trees averaged · main forecast",
               "Ridge + interactions": "straight-line effects, class × condition"}

# HUD palette: cyan for the new system, steel for the base, amber for the blended class.
CYAN, STEEL, AMBER = "#3ee0ff", "#9aa7b4", "#e3b93c"
HUD_LINE, HUD_GRID, HUD_DIM = "rgba(62,224,255,.28)", "rgba(62,224,255,.10)", "#7f93a6"
MONO = "'JetBrains Mono', Consolas, monospace"


# ---------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------
def with_profiles(cards):
    """Add each card's estimated capability profile (looked up by platform name) and the loss flag."""
    prof = [P.profile(n, c, D.CATALOG)[0] for n, c in zip(cards["platform"], cards["category"])]
    out = pd.concat([cards.reset_index(drop=True), pd.DataFrame(prof)], axis=1)
    out["lost"] = (out["p_status"] == D.STATUS_LOST).astype(float).where(out["p_status"].notna())
    return out


def _prep(df):
    X = df.reindex(columns=CAT_FEATS + NUM_FEATS).copy()
    for c in CAT_FEATS:
        X[c] = X[c].fillna("Unknown").astype(str).replace({"": "Unknown"})
    for c in NUM_FEATS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X


def _prep_lin(df):
    """Same inputs plus interaction columns, so the ridge can learn class- and link-specific effects."""
    X = _prep(df)
    for a, b in PAIRS:
        X[f"{a} x {b}"] = X[a] + " | " + X[b]
    return X


def _pipe(est, cats):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                             ("num", "passthrough", NUM_FEATS)])
    return Pipeline([("pre", pre), ("model", est)])


def _forest(classifier=False, trees=200):
    est = (RandomForestClassifier if classifier else RandomForestRegressor)(
        n_estimators=trees, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    return _pipe(est, CAT_FEATS)


@st.cache_resource(show_spinner="Training models on the current reports…", max_entries=4)
def _train(_key, _cards):
    cards = _cards
    groups = cards["id"]
    tr_ids, te_ids = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                          .split(cards, groups=groups))
    test_reports = set(groups.iloc[te_ids])
    is_te = groups.isin(test_reports)
    X, Xl = _prep(cards), _prep_lin(cards)
    fits, scores = {}, {}
    for tgt in OUTCOMES:
        y = cards[tgt]
        ok = y.notna()
        tr, te = ok & ~is_te, ok & is_te
        if tr.sum() < 30 or te.sum() < 10:
            continue
        if tgt == "lost":
            f = _forest(True, trees=120).fit(X[tr], y[tr].astype(int))
            p = f.predict_proba(X[te])[:, list(f.classes_).index(1)] if 1 in f.classes_ else np.zeros(te.sum())
            base = float(y[tr].mean())
            scores[tgt] = {"model": float(np.mean((p - y[te]) ** 2)), "baseline": float(np.mean((base - y[te]) ** 2)),
                           "metric": "Brier"}
        else:
            yv = y.astype(float)
            f = _forest(trees=200 if tgt == TARGET else 120).fit(X[tr], yv[tr])
            base = DummyRegressor(strategy="mean").fit(X[tr], yv[tr])
            scores[tgt] = {"model": mean_absolute_error(yv[te], f.predict(X[te])),
                           "baseline": mean_absolute_error(yv[te], base.predict(X[te])), "metric": "MAE"}
        fits[tgt] = f
    y = cards[TARGET].astype(float)
    tr, te = ~is_te, is_te
    three = {"Random forest": fits[TARGET],
             "Decision tree": _pipe(DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=SEED),
                                    CAT_FEATS).fit(X[tr], y[tr]),
             "Ridge + interactions": _pipe(Ridge(alpha=RIDGE_ALPHA), LIN_CATS).fit(Xl[tr], y[tr])}
    mae3 = {n: mean_absolute_error(y[te], np.clip(m.predict((Xl if n.startswith("Ridge") else X)[te]), 1, 5))
            for n, m in three.items()}
    mae3["Guess the average"] = scores[TARGET]["baseline"]
    return {"fits": fits, "three": three, "scores": scores, "mae": mae3, "n_train": int(tr.sum()),
            "n_test": int(te.sum()), "reports_test": len(test_reports)}


def _trees(model, rows):
    """Effectiveness forest per-tree predictions, shape (n_trees, n_rows)."""
    pipe = model["fits"][TARGET]
    Xt = pipe["pre"].transform(_prep(pd.DataFrame(rows)))
    return np.stack([t.predict(Xt) for t in pipe["model"].estimators_])


def _forecast(model, rows, alt_rows=None, w=0.0):
    """Effectiveness: forest mean and 10-90% tree spread per row, optionally blended with a second class."""
    t = _trees(model, rows)
    if alt_rows is not None and w > 0:
        t = (1 - w) * t + w * _trees(model, alt_rows)
    return t.mean(axis=0), np.percentile(t, 10, axis=0), np.percentile(t, 90, axis=0)


def _predict(model, tgt, rows, alt_rows=None, w=0.0):
    """Any outcome for many rows (loss = probability), blended the same way."""
    f = model["fits"].get(tgt)
    if f is None:
        return np.full(len(rows), np.nan)

    def one(rs):
        X = _prep(pd.DataFrame(rs))
        if tgt == "lost":
            cls = list(f.classes_)
            return f.predict_proba(X)[:, cls.index(1)] if 1 in cls else np.zeros(len(rs))
        return f.predict(X)
    v = one(rows)
    if alt_rows is not None and w > 0:
        v = (1 - w) * v + w * one(alt_rows)
    return v


def _profile_all(model, row, alt_row=None, w=0.0):
    return {t: float(_predict(model, t, [row], [alt_row] if alt_row else None, w)[0]) for t in OUTCOMES}


def _all_three(model, row, alt_row=None, w=0.0):
    """One effectiveness prediction from each of the three models for a single scenario."""
    out = {}
    for name in MODEL_NAMES:
        pipe = model["three"][name]
        prep = _prep_lin if name.startswith("Ridge") else _prep
        v = float(np.clip(pipe.predict(prep(pd.DataFrame([row])))[0], 1, 5))
        if alt_row is not None and w > 0:
            v = (1 - w) * v + w * float(np.clip(pipe.predict(prep(pd.DataFrame([alt_row])))[0], 1, 5))
        out[name] = v
    return out


def _distance(a, b):
    """Mean per-characteristic difference between two profiles, 0 (same) to 1 (opposite)."""
    d = []
    for k in P.ORDER:
        lv = P.ATTRS[k][1]
        d.append(abs(int(a[k]) - int(b[k])) / (len(lv) - 1) if lv else float(a[k] != b[k]))
    return sum(d) / len(d)


# ---------------------------------------------------------------------------
# 3D hologram (three.js), using the slimmed models embedded in analyst_base.html
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _models3d():
    path = Path(__file__).with_name("analyst_base.html")
    try:
        s = path.read_text(encoding="utf-8")
        i = s.index("const MODELS=") + len("const MODELS=")
        return json.JSONDecoder().raw_decode(s, i)[0]
    except (OSError, ValueError):
        return {}


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _glb_b64(name):
    """Base64 GLB for a platform (its own model first, then its category's), or None."""
    models = _models3d()
    for k in (_slug(name), _slug(D.PLATFORM_INFO.get(name, (name,))[0])):
        v = models.get(k)
        for _ in range(3):                      # 'ref' entries point at a model with the same geometry
            if v and v[0] == "ref":
                v = models.get(v[1])
        if v and v[0] == "glb":
            return v[1]
    return None


HOLO_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden;font-family:__MONO__;color:#cfe9f2}
#wrap{position:absolute;inset:0;border:1px solid rgba(62,224,255,.30);border-radius:8px;overflow:hidden;
  background:radial-gradient(ellipse at 50% 62%,rgba(62,224,255,.13),rgba(8,12,17,.95) 68%)}
#wrap::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:repeating-linear-gradient(0deg,rgba(255,255,255,.018) 0 1px,transparent 1px 3px)}
.c{position:absolute;width:22px;height:22px;border-color:#3ee0ff;border-style:solid;opacity:.9;z-index:2}
.tl{top:8px;left:8px;border-width:2px 0 0 2px}.tr{top:8px;right:8px;border-width:2px 2px 0 0}
.bl{bottom:8px;left:8px;border-width:0 0 2px 2px}.br{bottom:8px;right:8px;border-width:0 2px 2px 0}
.hud{position:absolute;font-size:11px;letter-spacing:.14em;text-transform:uppercase;pointer-events:none;z-index:2}
.k{color:#7f93a6}.v{color:#3ee0ff;font-weight:700}
.big{font-size:26px;color:#f0fbff;letter-spacing:.06em;font-weight:800;text-shadow:0 0 16px rgba(62,224,255,.55)}
#tl{top:22px;left:30px;line-height:1.75}#tr{top:22px;right:30px;text-align:right;line-height:1.75}
#bl{bottom:20px;left:30px;color:#7f93a6;font-size:10px;line-height:1.8}#br{bottom:20px;right:30px;text-align:right;color:#7f93a6;font-size:10px;line-height:1.8}
#msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#7f93a6;font-size:12px;letter-spacing:.2em}
canvas{position:absolute;inset:0}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#3ee0ff;margin-right:8px;
  box-shadow:0 0 8px #3ee0ff;animation:p 1.6s infinite}@keyframes p{50%{opacity:.25}}
.chip{display:inline-block;border:1px solid rgba(62,224,255,.45);border-radius:3px;padding:1px 7px;margin-left:6px;color:#3ee0ff}
.chip.a{border-color:rgba(227,185,60,.6);color:#e3b93c}
</style></head><body><div id="wrap"><div class="c tl"></div><div class="c tr"></div><div class="c bl"></div><div class="c br"></div>
<div id="msg">INITIALIZING 3D…</div>
<div class="hud" id="tl"><span class="dot"></span><span class="k">Target profile</span><br><span class="big">__NAME__</span><br>
<span class="k">Base</span> <span class="v">__BASE__</span> &nbsp; <span class="k">Class</span> <span class="v">__CLASS__</span><br>
<span class="k">Training data</span> <span class="v">__NREP__ reports in class</span></div>
<div class="hud" id="tr"><span class="k">Modifications</span><br>__MODS__</div>
<div class="hud" id="bl">__GHOSTNOTE__Shape: base system · modifications not drawn</div>
<div class="hud" id="br">Drag to rotate · scroll to zoom<br>Synthetic data · pipeline demo</div></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/"}}</script>
<script type="module">
const MAIN="__MAIN__", GHOST="__GHOST__";
const msg=document.getElementById("msg");
try{
 const THREE=await import("three");
 const {GLTFLoader}=await import("three/addons/loaders/GLTFLoader.js");
 const {OrbitControls}=await import("three/addons/controls/OrbitControls.js");
 const wrap=document.getElementById("wrap"), W=()=>wrap.clientWidth, H=()=>wrap.clientHeight;
 const r=new THREE.WebGLRenderer({antialias:true,alpha:true}); r.setPixelRatio(Math.min(2,devicePixelRatio));
 r.setSize(W(),H()); wrap.prepend(r.domElement);
 const scene=new THREE.Scene(), cam=new THREE.PerspectiveCamera(32,W()/H(),.01,1000);
 scene.fog=new THREE.Fog(0x080c11,6,13);
 scene.add(new THREE.HemisphereLight(0xbfefff,0x0b1218,1.3));
 const d=new THREE.DirectionalLight(0xffffff,1.5); d.position.set(3,5,4); scene.add(d);
 const rim=new THREE.DirectionalLight(0x3ee0ff,1.2); rim.position.set(-4,2,-3); scene.add(rim);
 const cyan=new THREE.Color(0x3ee0ff), amber=new THREE.Color(0xe3b93c);
 const bytes=b64=>Uint8Array.from(atob(b64),c=>c.charCodeAt(0)).buffer;
 const load=b64=>new Promise((ok,no)=>new GLTFLoader().parse(bytes(b64),"",g=>ok(g.scene),no));
 function fit(obj,size){const b=new THREE.Box3().setFromObject(obj),s=b.getSize(new THREE.Vector3()),c=b.getCenter(new THREE.Vector3());
   const k=size/Math.max(s.x,s.y,s.z); obj.scale.setScalar(k); obj.position.sub(c.multiplyScalar(k)); obj.position.y+=s.y*k/2; return s.y*k;}
 const fades=[];
 function holo(root,tint,solid){root.traverse(o=>{if(!o.isMesh)return; o.geometry.computeVertexNormals();
   const base=(o.material&&o.material.color)?o.material.color.clone():new THREE.Color(0x8899aa);
   o.material=solid?new THREE.MeshStandardMaterial({color:base.lerp(tint,.25),flatShading:true,metalness:.3,roughness:.5,
       emissive:tint,emissiveIntensity:.07,transparent:true,opacity:0})
     :new THREE.MeshBasicMaterial({color:tint,transparent:true,opacity:0});
   const e=new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry,28),new THREE.LineBasicMaterial({color:tint,transparent:true,opacity:0}));
   o.add(e); fades.push([o.material,solid?.93:.06],[e.material,solid?.35:.6]);});}
 const grp=new THREE.Group(); scene.add(grp);
 const m=await load(MAIN); holo(m,cyan,true); const h=fit(m,2.6); grp.add(m);
 let g=null; if(GHOST){g=await load(GHOST); holo(g,amber,false); fit(g,1.1); g.position.x+=2.05; g.position.z-=.7; grp.add(g);}
 // floor: grid, pulsing ring, rotating tick ring, rising particles
 const grid=new THREE.GridHelper(10,40,0x3ee0ff,0x3ee0ff); grid.material.transparent=true; grid.material.opacity=.12; scene.add(grid);
 const ring=new THREE.Mesh(new THREE.RingGeometry(1.55,1.59,128),new THREE.MeshBasicMaterial({color:0x3ee0ff,transparent:true,opacity:.5,side:THREE.DoubleSide}));
 ring.rotation.x=-Math.PI/2; ring.position.y=.003; scene.add(ring);
 const pulse=new THREE.Mesh(new THREE.RingGeometry(.98,1.0,128),new THREE.MeshBasicMaterial({color:0x3ee0ff,transparent:true,opacity:.6,side:THREE.DoubleSide}));
 pulse.rotation.x=-Math.PI/2; pulse.position.y=.004; scene.add(pulse);
 const tk=[]; for(let i=0;i<120;i++){const a=i/120*Math.PI*2,l=(i%10==0)?.16:.07;
   tk.push(Math.cos(a)*1.85,0,Math.sin(a)*1.85,Math.cos(a)*(1.85+l),0,Math.sin(a)*(1.85+l));}
 const tg=new THREE.BufferGeometry(); tg.setAttribute("position",new THREE.Float32BufferAttribute(tk,3));
 const ticks=new THREE.LineSegments(tg,new THREE.LineBasicMaterial({color:0x3ee0ff,transparent:true,opacity:.5})); ticks.position.y=.005; scene.add(ticks);
 const N=260,pp=new Float32Array(N*3),sp=new Float32Array(N);
 for(let i=0;i<N;i++){const a=Math.random()*Math.PI*2,rr=.4+Math.random()*2.6; pp[i*3]=Math.cos(a)*rr; pp[i*3+1]=Math.random()*3; pp[i*3+2]=Math.sin(a)*rr; sp[i]=.08+Math.random()*.25;}
 const pg=new THREE.BufferGeometry(); pg.setAttribute("position",new THREE.BufferAttribute(pp,3));
 const pts=new THREE.Points(pg,new THREE.PointsMaterial({color:0x3ee0ff,size:.022,transparent:true,opacity:.55})); scene.add(pts);
 const scan=new THREE.Mesh(new THREE.PlaneGeometry(3.6,3.6),new THREE.MeshBasicMaterial({color:0x3ee0ff,transparent:true,opacity:.08,side:THREE.DoubleSide}));
 scan.rotation.x=-Math.PI/2; scene.add(scan);
 const wide=W()/H()>2; cam.position.set(wide?3.1:3.6,1.75,wide?3.3:3.9);
 const ctl=new OrbitControls(cam,r.domElement); ctl.target.set(0,h*.40,0);
 ctl.enableDamping=true; ctl.autoRotate=true; ctl.autoRotateSpeed=.9; ctl.enablePan=false; ctl.minDistance=2.2; ctl.maxDistance=9;
 msg.remove();
 // boot-up: the model materialises bottom-up behind a bright scan plane
 const clip=new THREE.Plane(new THREE.Vector3(0,-1,0),0); r.localClippingEnabled=true;
 fades.forEach(([mat])=>{mat.clippingPlanes=[clip];});
 const clk=new THREE.Clock(), BOOT=1.6;
 (function tick(){const t=clk.getElapsedTime(), k=Math.min(1,t/BOOT), e=1-Math.pow(1-k,3);
   clip.constant=e*h*1.15+.001; fades.forEach(([mat,o])=>{mat.opacity=o*Math.min(1,k*1.6);});
   scan.position.y=k<1?clip.constant:(Math.sin((t-BOOT)*.9)*.5+.5)*h*1.05; scan.material.opacity=k<1?.22:.08;
   ring.material.opacity=.32+.16*Math.sin(t*2); ticks.rotation.y=-t*.12;
   const ps=(t*.45)%1; pulse.scale.setScalar(1+ps*1.6); pulse.material.opacity=.55*(1-ps);
   const a=pg.attributes.position.array; for(let i=0;i<N;i++){a[i*3+1]+=sp[i]*.006; if(a[i*3+1]>3)a[i*3+1]=0;} pg.attributes.position.needsUpdate=true;
   if(g) g.rotation.y=t*.4;
   ctl.update(); r.render(scene,cam); requestAnimationFrame(tick);})();
 new ResizeObserver(()=>{r.setSize(W(),H()); cam.aspect=W()/H(); cam.updateProjectionMatrix();}).observe(wrap);
}catch(e){msg.textContent="3D VIEW UNAVAILABLE (needs internet for three.js)";}
</script></body></html>"""

HOLO_HEIGHT = 470


def _hologram(name, base, base_cat, n_cat, mods, alt_cat):
    main = _glb_b64(base)
    if not main:
        return False
    ghost = _glb_b64(alt_cat) if alt_cat else None
    esc = lambda s: html.escape(str(s))
    mods_html = "<br>".join(f"<span class='chip{' a' if k == 'Behaves like' else ''}'>{esc(k)}: {esc(v)}</span>"
                            for k, v in mods)
    page = (HOLO_HTML.replace("__MONO__", MONO).replace("__NAME__", esc(name or "New system"))
            .replace("__BASE__", esc(base)).replace("__CLASS__", esc(base_cat)).replace("__NREP__", str(n_cat))
            .replace("__MODS__", mods_html or "<span class='chip'>None · stock build</span>")
            .replace("__GHOSTNOTE__", f"<span style='color:{AMBER}'>■</span> amber wireframe: behaves partly like "
                     f"{esc(alt_cat)}<br>" if ghost else "")
            .replace("__MAIN__", main).replace("__GHOST__", ghost or ""))
    st.iframe(page, height=HOLO_HEIGHT)
    return True


# ---------------------------------------------------------------------------
# HUD styling
# ---------------------------------------------------------------------------
HUD_CSS = f"""
.hud-h {{ font-family:{MONO}; font-size:.78rem; letter-spacing:.16em; color:{CYAN}; text-transform:uppercase;
  margin:18px 0 6px; display:flex; align-items:center; gap:10px; }}
.hud-h::after {{ content:""; flex:1; height:1px; background:linear-gradient(90deg,{HUD_LINE},transparent); }}
.hud-h b {{ color:{HUD_DIM}; font-weight:400; }}
[class*="st-key-hud_"] {{ border:1px solid {HUD_LINE}; border-radius:6px; padding:12px 16px 14px;
  background:linear-gradient(180deg,rgba(62,224,255,.045),rgba(15,19,24,0) 60%); position:relative; }}
[class*="st-key-hud_"]::before, [class*="st-key-hud_"]::after {{ content:""; position:absolute; width:12px; height:12px;
  border-color:{CYAN}; border-style:solid; opacity:.8; pointer-events:none; }}
[class*="st-key-hud_"]::before {{ top:-1px; left:-1px; border-width:2px 0 0 2px; border-top-left-radius:6px; }}
[class*="st-key-hud_"]::after {{ bottom:-1px; right:-1px; border-width:0 2px 2px 0; border-bottom-right-radius:6px; }}
.ro {{ font-family:{MONO}; }}
.ro .lbl {{ font-size:.7rem; letter-spacing:.14em; color:{HUD_DIM}; text-transform:uppercase; }}
.ro .num {{ font-size:2.5rem; font-weight:800; line-height:1.05; }}
.ro .of {{ font-size:1rem; color:{HUD_DIM}; }}
.ro .sub {{ font-size:.78rem; color:{HUD_DIM}; }}
.bar {{ height:6px; border-radius:3px; background:rgba(255,255,255,.07); margin:8px 0 4px; position:relative; overflow:hidden; }}
.bar > i {{ position:absolute; left:0; top:0; bottom:0; border-radius:3px; }}
.bar > s {{ position:absolute; top:-2px; bottom:-2px; width:2px; background:#fff; opacity:.8; }}
.pod {{ border:1px solid {HUD_GRID}; border-radius:6px; padding:10px 12px; background:rgba(62,224,255,.03); font-family:{MONO}; }}
.pod .t {{ font-size:.72rem; letter-spacing:.12em; color:{CYAN}; text-transform:uppercase; }}
.pod .v {{ font-size:1.6rem; font-weight:800; }}
.pod .d {{ font-size:.72rem; color:{HUD_DIM}; }}
.tile {{ font-family:{MONO}; padding:2px 4px; }}
.tile .lbl {{ font-size:.68rem; letter-spacing:.14em; color:{HUD_DIM}; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.tile .val {{ font-size:2.2rem; font-weight:800; line-height:1.1; }}
.tile .sub {{ font-size:.74rem; color:{HUD_DIM}; }}
.cond-line {{ font-family:{MONO}; font-size:.72rem; letter-spacing:.12em; color:{HUD_DIM}; border-top:1px solid {HUD_GRID};
  padding-top:8px; margin-top:4px; }}
.mods {{ font-family:{MONO}; display:flex; flex-direction:column; gap:4px; }}
.mod {{ display:flex; gap:14px; align-items:baseline; padding:6px 10px; border-left:2px solid {HUD_GRID};
  background:rgba(62,224,255,.025); }}
.mod .pts {{ min-width:86px; font-weight:800; font-size:1.05rem; }}
.mod .what {{ color:#cfd8e1; font-size:.85rem; }}
.mod.tot {{ border-left-color:{CYAN}; background:rgba(62,224,255,.07); }}
.hangar {{ width:100%; border-collapse:collapse; font-family:{MONO}; font-size:.82rem; }}
.hangar th {{ text-align:left; color:{HUD_DIM}; font-weight:400; font-size:.68rem; letter-spacing:.12em;
  text-transform:uppercase; border-bottom:1px solid {HUD_LINE}; padding:4px 6px; }}
.hangar td {{ padding:7px 6px; border-bottom:1px solid {HUD_GRID}; vertical-align:middle; }}
.hangar .rk {{ color:{CYAN}; font-weight:800; width:34px; }}
.hangar .n {{ font-weight:800; font-size:1rem; text-align:right; min-width:64px; white-space:nowrap; }}
.hangar tr.tot td {{ border-top:1px solid {CYAN}; background:rgba(62,224,255,.06); }}
.hangar .sm {{ color:{HUD_DIM}; font-size:.72rem; }}
.verdict {{ font-family:{MONO}; font-size:.8rem; letter-spacing:.1em; padding:6px 10px; border-radius:4px; display:inline-block; }}
"""


def _h(num, title, note=""):
    st.html(f"<div class='hud-h'>// {num:02d} &nbsp;{html.escape(title)}"
            + (f" <b>{html.escape(note)}</b>" if note else "") + "</div>")


def _bar(v, color, lo=None, hi=None, mark=None):
    pct = lambda x: max(0, min(100, (x - 1) / 4 * 100))
    rng = (f"<i style='left:{pct(lo):.1f}%;width:{pct(hi) - pct(lo):.1f}%;background:{color};opacity:.25'></i>"
           if lo is not None else "")
    mk = f"<s style='left:{pct(mark):.1f}%'></s>" if mark is not None else ""
    return (f"<div class='bar'>{rng}<i style='width:{pct(v):.1f}%;background:{color};"
            f"box-shadow:0 0 10px {color}'></i>{mk}</div>")


def _readout(label, v, color, sub, lo=None, hi=None, mark=None):
    return (f"<div class='ro'><div class='lbl'>{html.escape(label)}</div>"
            f"<div class='num' style='color:{color};text-shadow:0 0 18px {color}55'>{v:.1f}<span class='of'> / 5</span></div>"
            f"{_bar(v, color, lo, hi, mark)}<div class='sub'>{sub}</div></div>")


def _eff_color(v):
    return D.SCALE_5[int(min(4, max(0, round(v) - 1)))]


def _hud_fig(fig, height):
    style(fig, height=height)
    fig.update_layout(font=dict(family="JetBrains Mono, Consolas, monospace", color=FG, size=12),
                      legend=dict(bgcolor="rgba(8,12,17,.7)", bordercolor=HUD_LINE, borderwidth=1))
    fig.update_xaxes(gridcolor=HUD_GRID, zerolinecolor=HUD_GRID, linecolor=HUD_LINE)
    fig.update_yaxes(gridcolor=HUD_GRID, zerolinecolor=HUD_GRID, linecolor=HUD_LINE)
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
PRESETS = {
    "Quiet day": {"lab_ew": "None", "lab_ad": "None", "lab_tod": "Day", "lab_wx": "Clear"},
    "Night + EW": {"lab_ew": "High", "lab_ad": "Medium", "lab_tod": "Night", "lab_wx": "Overcast"},
    "Contested": {"lab_ew": "Medium", "lab_ad": "High", "lab_tod": "Dawn / dusk", "lab_wx": "Rain"},
    "Worst case": {"lab_ew": "High", "lab_ad": "High", "lab_tod": "Night", "lab_wx": "Fog"},
}
HANGAR_MAX = 8


def _apply_preset(name):
    st.session_state.update(PRESETS[name])


def _design_rows(d, cond):
    """(row, alt_row, w) for a saved or current design under the given conditions."""
    row = {"category": d["base_cat"], "origin": d["origin"], "p_field_modified": int(d["field_mod"]),
           **d["profile"], **cond}
    alt = {**row, "category": d["alt_cat"]} if d["alt_cat"] else None
    return row, alt, d["w"]


def _fmt(tgt, v):
    return f"{v:.0%}" if tgt == "lost" else f"{v:.1f}"


def _good(tgt, delta):
    """+1 better, -1 worse, 0 about the same."""
    thr = 0.03 if tgt == "lost" else 0.05
    if abs(delta) < thr:
        return 0
    return 1 if (delta > 0) == OUTCOMES[tgt][2] else -1


def _status_tile(label, value, color, sub=""):
    return (f"<div class='tile'><div class='lbl'>{html.escape(label)}</div>"
            f"<div class='val' style='color:{color};text-shadow:0 0 16px {color}55'>{value}</div>"
            f"<div class='sub'>{sub}</div></div>")


def _delta_html(tgt, delta):
    g = _good(tgt, delta)
    col = D.GREEN if g > 0 else D.RED if g < 0 else HUD_DIM
    big = abs(delta) >= (0.03 if tgt == "lost" else 0.05)
    sign = ("▲" if delta > 0 else "▼") if big else "■"
    txt = f"{delta * 100:+.0f} pts" if tgt == "lost" else f"{delta:+.2f}"
    return f"<span style='color:{col};font-weight:800'>{sign} {txt}</span>"


def _outcome_color(tgt, v):
    return _eff_color(5 - v * 8) if tgt == "lost" else _eff_color(v)   # 0% loss -> green, 50%+ -> red


def render(conn):
    raw = D.platform_rows_df(conn)
    raw = raw[(raw["rated_individually"] == 1) & raw["p_effectiveness"].notna()]
    if len(raw) < MIN_CARDS:
        st.info(f"The Lab trains models on the reports, so it needs at least {MIN_CARDS} rated platform cards "
                f"(there are {len(raw)}). Add reports, or generate sample data above.", icon=":material/science:")
        return False
    cards = with_profiles(raw)

    st.html(f"<style>{HUD_CSS}</style>")
    key = hashlib.md5(cards[["id"] + list(OUTCOMES) + CAT_FEATS + NUM_FEATS].to_csv(index=False).encode()).hexdigest()
    model = _train(key, cards)
    synthetic = bool((cards["notes"].fillna("") == "[sample data]").all())

    st.html("<div class='caveat'><span>⚠</span><span>Forecasts from the reports, for exploring ideas. "
            + ("Everything here is <b>synthetic sample data with planted patterns</b>: this shows the method "
               "working, it says nothing about real weapons. " if synthetic else
               "Some or all of these reports are synthetic sample data; treat the numbers as a method demo. ")
            + "Platform characteristics are <b>illustrative estimates</b>. The models only know a system's class, "
              "origin, field-modified flag, estimated characteristics and the conditions below.</span></div>")

    counts = cards["platform"].value_counts()
    known = sorted(counts.index, key=lambda n: (D.PLATFORM_INFO.get(n, ("~",))[0], n))
    default = "HAWK" if "HAWK" in counts else counts.index[0]

    hero = st.container()          # filled after the controls below are read
    status = st.container()
    left, right = st.columns([5, 7], gap="large")

    # ------------------------------------------------------------- controls
    with left:
        _h(1, "Base platform", "closest known system")
        with st.container(key="hud_base"):
            base = st.selectbox("Closest existing system", known, index=known.index(default), key="lab_base",
                                format_func=lambda n: f"{n}  ·  {counts[n]} report{'s' if counts[n] != 1 else ''}")
            row0 = cards[cards["platform"] == base].iloc[0]
            base_cat, base_origin = row0["category"], row0["origin"]
            base_prof, est = P.profile(base, base_cat, D.CATALOG)
            n_cat = int((cards["category"] == base_cat).sum())
            st.caption(f"Class **{base_cat}** · {n_cat} reports in this class · origin **{base_origin}**")
            name = st.text_input("Designation for the new system", value=f"{base}-X", key=f"lab_name_{base}")
            with st.expander("Base capability profile (estimates)", icon=":material/fact_check:"):
                st.html("<table class='hangar'>" + "".join(
                    f"<tr><td>{html.escape(P.ATTRS[k][0])}</td><td><b>{html.escape(P.label(k, base_prof[k]))}</b>"
                    f"</td><td class='sm'>ESTIMATE</td></tr>" for k in P.ORDER) + "</table>")
                st.caption(("Coarse estimates from public descriptions of the system; illustrative, not "
                            "authoritative." if est else
                            "Not in the profile table: these are the class's most common values.")
                           + " Replace with sourced values (e.g. ODIN WEG) before any real use.")

        _h(2, "Design", "starts at base estimates")
        with st.container(key="hud_design"):
            prof = {}
            c1, c2 = st.columns(2)
            for i, k in enumerate(P.ORDINAL):
                lv = P.ATTRS[k][1]
                col = c1 if i % 2 == 0 else c2
                prof[k] = lv.index(col.select_slider(
                    P.ATTRS[k][0], lv, value=lv[int(base_prof[k])], key=f"lab_{k}_{base}", help=P.ATTRS[k][2]))
            for k in P.CATEGORICAL:
                opts = P.CHOICES[k]
                col = c1 if k == "mobility" else c2
                prof[k] = col.selectbox(P.ATTRS[k][0], opts, index=opts.index(base_prof[k]),
                                        key=f"lab_{k}_{base}", help=P.ATTRS[k][2])
            changed = [k for k in P.ORDER if prof[k] != base_prof[k]]
            if changed:
                st.caption("Changed from base: " + ", ".join(
                    f"**{P.ATTRS[k][0]}** {P.label(k, base_prof[k])} → {P.label(k, prof[k])}" for k in changed))

        _h(3, "Class and build")
        with st.container(key="hud_mods"):
            origin_opts = [base_origin] + [o for o in ORIGINS if o != base_origin]
            origin = st.segmented_control("Built / supplied by", origin_opts, default=base_origin,
                                          key=f"lab_origin_{base}") or base_origin
            field_mod = st.toggle("Field-modified build", key="lab_fieldmod",
                                  help="Same question as the report form: was the system modified in the field?")
            other_classes = ["None"] + sorted(c for c in cards["category"].unique() if c != base_cat)
            alt_cat = st.selectbox("Also behaves like another class", other_classes, key=f"lab_alt_{base}",
                                   help="Example: a HAWK with a drone-style data link might behave partly like "
                                        "Counter-UAS or a UAS class.")
            w = 0.0
            if alt_cat != "None":
                w = st.slider(f"How much like {alt_cat}", 0, 100, 40, step=10, format="%d%%", key="lab_w") / 100
                st.caption("Analyst assumption: the forecasts blend the two classes' learned behaviour in this "
                           "ratio. The models have not seen this hybrid.")

        _h(4, "Mission conditions")
        with st.container(key="hud_cond"):
            for k, v in PRESETS["Quiet day"].items():
                st.session_state.setdefault(k, "Low" if k == "lab_ad" else v)
            st.caption("Mission presets")
            pc = st.columns(len(PRESETS))
            for c, pname in zip(pc, PRESETS):
                c.button(pname, key=f"lab_preset_{pname}", on_click=_apply_preset, args=(pname,), width="stretch")
            cond = {
                "ew_level": st.select_slider("Enemy EW / jamming", LEVELS, key="lab_ew"),
                "ad_level": st.select_slider("Enemy air defense", LEVELS, key="lab_ad"),
                "time_of_day": st.segmented_control("Time of day", D.values(D.TIME_OF_DAY), key="lab_tod") or "Day",
                "weather": st.select_slider("Weather", D.values(D.WEATHER), key="lab_wx"),
                "terrain": st.selectbox("Terrain", D.values(D.TERRAIN), key="lab_terrain"),
            }
            with st.expander("More conditions (unit, task, enemy)"):
                cond["unit_type"] = st.selectbox("Unit type", D.UNIT_TYPES, key="lab_unit")
                cond["task"] = st.selectbox("Primary task", D.TASKS, index=D.TASKS.index("Defense"), key="lab_task")
                cond["enemy_type"] = st.selectbox("Enemy type", D.ENEMY_TYPES, key="lab_etype")
                cond["enemy_strength"] = st.select_slider("Enemy strength", D.values(D.ENEMY_STRENGTH), key="lab_estr")
                cond["enemy_adaptation"] = st.selectbox("Enemy adaptation", D.values(D.ENEMY_ADAPT), key="lab_eadapt")

    # ------------------------------------------------------------- forecasts
    has_alt = alt_cat != "None"
    design = {"name": name or "New system", "base": base, "base_cat": base_cat, "origin": origin,
              "field_mod": bool(field_mod), "alt_cat": alt_cat if has_alt else None, "w": w, "profile": dict(prof)}
    base_design = {**design, "origin": base_origin, "field_mod": False, "alt_cat": None, "w": 0.0,
                   "profile": dict(base_prof)}
    new_row, alt_row, _ = _design_rows(design, cond)
    ref_row, _, _ = _design_rows(base_design, cond)
    blend = (lambda rows: [{**r, "category": alt_cat} for r in rows]) if has_alt else (lambda rows: None)

    m_new, lo_new, hi_new = (float(a[0]) for a in _forecast(model, [new_row], blend([new_row]), w))
    m_ref = float(_forecast(model, [ref_row])[0][0])
    out_new = _profile_all(model, new_row, alt_row, w)
    out_ref = _profile_all(model, ref_row)
    out_new[TARGET], out_ref[TARGET] = m_new, m_ref
    three = _all_three(model, new_row, alt_row, w)
    spread = max(three.values()) - min(three.values())
    agree = spread < AGREE

    # nearest reported systems by profile (+ same class bonus), and the extrapolation alert
    rep = cards.groupby("platform").agg(n=("id", "size"), category=("category", "first")).reset_index()
    rep["dist"] = [_distance(prof, P.profile(n, c, D.CATALOG)[0]) for n, c in zip(rep["platform"], rep["category"])]
    rep = rep.sort_values(["dist", "n"], ascending=[True, False])
    near = rep.head(3)
    far = float(near["dist"].iloc[0]) > FAR

    # ------------------------------------------------------------- hero + status bar
    with hero:
        mods = [(P.ATTRS[k][0].split(" /")[0].split(" (")[0], P.label(k, prof[k])) for k in changed]
        if origin != base_origin:
            mods.append(("Origin", origin))
        if field_mod:
            mods.append(("Build", "field-modified"))
        if has_alt:
            mods.append(("Behaves like", alt_cat))
        if not _hologram(name, base, base_cat, n_cat, mods, alt_cat if has_alt else None):
            st.caption("No 3D model for this system.")
    with status, st.container(key="hud_status"):
        t = st.columns(5)
        t[0].html(_status_tile(f"{(name or 'New system')[:24]} · effect.", f"{m_new:.1f}", _eff_color(m_new),
                               f"80% of trees {lo_new:.1f}–{hi_new:.1f}"))
        t[1].html(_status_tile(f"{base} as fielded", f"{m_ref:.1f}", STEEL, f"change {_delta_html(TARGET, m_new - m_ref)}"))
        s_new, s_ref = out_new["p_rating_survivability"], out_ref["p_rating_survivability"]
        t[2].html(_status_tile("Platform survival", _fmt("p_rating_survivability", s_new),
                               _outcome_color("p_rating_survivability", s_new),
                               f"vs base {_delta_html('p_rating_survivability', s_new - s_ref)}"))
        l_new, l_ref = out_new["lost"], out_ref["lost"]
        t[3].html(_status_tile("Loss risk", _fmt("lost", l_new), _outcome_color("lost", l_new),
                               f"vs base {_delta_html('lost', l_new - l_ref)}"))
        t[4].html(_status_tile("Three models", "AGREE" if agree else "SPLIT", D.GREEN if agree else D.ORANGE,
                               f"spread {spread:.2f} pts"))
        line = (f"CONDITIONS · EW {cond['ew_level']} · AIR DEFENSE {cond['ad_level']} · {cond['time_of_day']} · "
                f"{cond['weather']} · {cond['terrain']}")
        if far:
            line += f"   <span style='color:{D.ORANGE}'>▲ EXTRAPOLATION: no reported system has a profile like this</span>"
        st.html(f"<div class='cond-line'>{line}</div>")

    with right:
        # ---------------- trade-off profile
        _h(5, "Trade-off profile", "new design vs base, every outcome the form collects")
        with st.container(key="hud_profile"):
            rows_html = ""
            for tgt, (lbl, short, hib) in OUTCOMES.items():
                vn, vr = out_new[tgt], out_ref[tgt]
                if np.isnan(vn):
                    continue
                bar_v = (lambda v: 1 + 4 * min(1, v / 0.5)) if tgt == "lost" else (lambda v: v)
                rows_html += (f"<tr><td><b>{lbl}</b><br><span class='sm'>{'lower is better' if not hib else '1–5'}"
                              f"</span></td><td style='width:30%'>{_bar(bar_v(vr), STEEL)}"
                              f"<span class='sm'>base {_fmt(tgt, vr)}</span></td>"
                              f"<td style='width:30%'>{_bar(bar_v(vn), _outcome_color(tgt, vn))}"
                              f"<span class='sm'>new {_fmt(tgt, vn)}</span></td>"
                              f"<td class='n'>{_delta_html(tgt, vn - vr)}</td></tr>")
            st.html(f"<table class='hangar'><tr><th>Outcome</th><th>{html.escape(base)}</th>"
                    f"<th>{html.escape(name or 'New')}</th><th>Change</th></tr>{rows_html}</table>")
            st.caption("Each outcome has its own model, trained on the reports that answered it. Green ▲ = better, "
                       "red ▼ = worse. A forecast of what units would report, under the conditions set.")

        # ---------------- stat modifiers
        _h(6, "Stat modifiers", "what each change does, one at a time")
        with st.container(key="hud_modifiers"):
            steps, row = [], dict(ref_row)
            prev = {t: out_ref[t] for t in (TARGET, "p_rating_survivability", "lost")}

            def step(label, new_row_):
                vals = {t: float(_predict(model, t, [new_row_])[0]) if t != TARGET
                        else float(_forecast(model, [new_row_])[0][0]) for t in prev}
                steps.append((label, {t: vals[t] - prev[t] for t in prev}))
                prev.update(vals)
            for k in changed:
                row = {**row, k: prof[k]}
                step(f"{P.ATTRS[k][0]}: {P.label(k, prof[k])}", row)
            if origin != base_origin:
                row = {**row, "origin": origin}
                step(f"Origin: {origin}", row)
            if field_mod:
                row = {**row, "p_field_modified": 1}
                step("Field-modified build", row)
            if has_alt and w > 0:
                steps.append((f"Behaves {int(w * 100)}% like {alt_cat}",
                              {TARGET: m_new - prev[TARGET],
                               "p_rating_survivability": out_new["p_rating_survivability"] - prev["p_rating_survivability"],
                               "lost": out_new["lost"] - prev["lost"]}))
            body = "".join(f"<tr><td>{html.escape(lbl)}</td>" + "".join(
                f"<td class='n'>{_delta_html(t, d[t])}</td>" for t in (TARGET, "p_rating_survivability", "lost"))
                + "</tr>" for lbl, d in steps)
            if not body:
                body = "<tr><td colspan=4 class='sm'>Stock build: change a characteristic or the build to see modifiers.</td></tr>"
            st.html(f"<table class='hangar'><tr><th>Change</th><th>Effect.</th><th>Survival</th><th>Loss risk</th></tr>"
                    f"<tr><td class='sm'>{html.escape(base)} as fielded</td><td class='n'>{m_ref:.2f}</td>"
                    f"<td class='n'>{out_ref['p_rating_survivability']:.2f}</td><td class='n'>{out_ref['lost']:.0%}</td></tr>"
                    f"{body}<tr class='tot'><td><b>{html.escape(name or 'New system')}</b></td>"
                    f"<td class='n'>{m_new:.2f}</td><td class='n'>{out_new['p_rating_survivability']:.2f}</td>"
                    f"<td class='n'>{out_new['lost']:.0%}</td></tr></table>")
            st.caption("Each line adds one change to the line above and shows how much the forecasts move. The "
                       "order matters a little, because effects can interact.")

        # ---------------- trade space
        _h(7, "Trade space", "every reported system under these conditions")
        plats = rep[rep["n"] >= 3]
        prow = [{"category": c, "origin": D.PLATFORM_INFO.get(n, (c, "Custom"))[1], "p_field_modified": 0,
                 **P.profile(n, c, D.CATALOG)[0], **cond} for n, c in zip(plats["platform"], plats["category"])]
        if prow:
            px_ = _predict(model, "p_rating_survivability", prow)
            py_ = _forecast(model, prow)[0]
            cmap = {c: col for c, col in zip(sorted(cards["category"].unique()),
                                             ["#4a9bd1", "#e3b93c", "#46b673", "#ec8a3c", "#b07fd6", "#5fc7c2",
                                              "#e05d5d", "#a3c94a", "#d987b6", "#8d99a6", "#7fa7e0", "#c9a36b",
                                              "#6fbf9a", "#ffffff"])}
            fig3 = go.Figure()
            for c in sorted(plats["category"].unique()):
                m = (plats["category"] == c).to_numpy()
                fig3.add_trace(go.Scatter(
                    x=px_[m], y=py_[m], mode="markers", name=c, opacity=.75,
                    marker=dict(size=8 + 6 * np.sqrt(plats["n"].to_numpy()[m] / plats["n"].max()), color=cmap.get(c)),
                    text=plats["platform"].to_numpy()[m],
                    hovertemplate="%{text}<br>survival %{x:.2f} · effectiveness %{y:.2f}<extra></extra>"))
            fig3.add_trace(go.Scatter(x=[out_ref["p_rating_survivability"], out_new["p_rating_survivability"]],
                                      y=[m_ref, m_new], mode="lines", line=dict(color=CYAN, width=2, dash="dot"),
                                      showlegend=False, hoverinfo="skip"))
            fig3.add_trace(go.Scatter(x=[out_ref["p_rating_survivability"]], y=[m_ref], mode="markers+text",
                                      name=f"{base} as fielded", text=[base], textposition="bottom center",
                                      marker=dict(size=16, color=STEEL, symbol="circle-open", line=dict(width=3))))
            fig3.add_trace(go.Scatter(x=[out_new["p_rating_survivability"]], y=[m_new], mode="markers+text",
                                      name=name or "New system", text=[name or "New"], textposition="top center",
                                      textfont=dict(color=CYAN),
                                      marker=dict(size=20, color=CYAN, symbol="diamond",
                                                  line=dict(color="#ffffff", width=1.5))))
            fig3.update_xaxes(title="Predicted platform survival →", range=[1, 5])
            fig3.update_yaxes(title="Predicted effectiveness →", range=[1, 5])
            fig3.update_layout(legend=dict(font=dict(size=10)))
            st.plotly_chart(_hud_fig(fig3, 430), width="stretch", config={"displayModeBar": False})
            st.caption("Each dot is a reported platform, forecast under the conditions you set (size = number of "
                       "reports). Top right is best. The dotted line shows how your changes move the design.")

        # ---------------- nearest systems
        _h(8, "Nearest real systems", "the reports behind this forecast")
        with st.container(key="hud_near"):
            cols = st.columns(3)
            for col, (_, r) in zip(cols, near.iterrows()):
                sim = max(0.0, 1 - float(r["dist"]))
                col.html(f"<div class='pod'><div class='t'>{html.escape(r['platform'])}</div>"
                         f"<div class='v' style='color:{CYAN}'>{sim:.0%}</div>{_bar(1 + 4 * sim, CYAN)}"
                         f"<div class='d'>profile match · {int(r['n'])} reports<br>{html.escape(r['category'])}</div></div>")
            if far:
                st.html(f"<div class='verdict' style='color:{D.ORANGE};border:1px solid {D.ORANGE}66;"
                        f"background:{D.ORANGE}14'>▲ EXTRAPOLATION · the closest reported profile is only "
                        f"{1 - float(near['dist'].iloc[0]):.0%} similar: the models are guessing beyond the data</div>")
            else:
                st.caption("Profile match compares the seven estimated characteristics. Open these systems' reports "
                           "in the Report database to check the forecast against what units said.")

        # ---------------- three models
        _h(9, "Three models", "independent effectiveness forecasts")
        with st.container(key="hud_three"):
            cols = st.columns(3)
            for col, nm in zip(cols, MODEL_NAMES):
                v = three[nm]
                col.html(f"<div class='pod'><div class='t'>{nm}</div>"
                         f"<div class='v' style='color:{_eff_color(v)}'>{v:.1f}</div>{_bar(v, _eff_color(v))}"
                         f"<div class='d'>{MODEL_BLURB[nm]}<br>avg miss on unseen reports "
                         f"{model['mae'][nm]:.2f}</div></div>")
            col = D.GREEN if agree else D.ORANGE
            st.html(f"<div class='verdict' style='color:{col};border:1px solid {col}66;background:{col}14'>"
                    + (f"▲ MODELS AGREE · spread {spread:.2f} pts" if agree else
                       f"▲ MODELS DISAGREE · spread {spread:.2f} pts: treat this forecast as uncertain")
                    + "</div>")

        # ---------------- stress test
        _h(10, "Stress test", "enemy jamming from none to high")
        sweep = lambda r: [{**r, "ew_level": lv} for lv in LEVELS]
        fig = go.Figure()
        series = [(name or "New system", _forecast(model, sweep(new_row), blend(sweep(new_row)), w), CYAN),
                  (f"{base} as fielded", _forecast(model, sweep(ref_row)), STEEL)]
        if has_alt:
            series.append((f"Typical {alt_cat}", _forecast(model, sweep({**ref_row, "category": alt_cat})), AMBER))
        for label, (m, lo, hi), colr in series:
            fig.add_trace(go.Scatter(x=LEVELS + LEVELS[::-1], y=list(hi) + list(lo[::-1]), fill="toself",
                                     fillcolor=colr, opacity=.12, line=dict(width=0), hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=LEVELS, y=m, name=label, mode="lines+markers",
                                     line=dict(color=colr, width=3, shape="spline"),
                                     marker=dict(size=9, symbol="diamond", line=dict(color="#0f1318", width=1)),
                                     hovertemplate="%{x}: %{y:.2f}<extra>" + html.escape(label) + "</extra>"))
        fig.add_vline(x=LEVELS.index(cond["ew_level"]), line=dict(color=CYAN, dash="dot", width=1))
        fig.update_yaxes(range=[1, 5], title="Predicted effectiveness", dtick=1)
        fig.update_xaxes(title="Enemy EW / jamming")
        st.plotly_chart(_hud_fig(fig, 320), width="stretch", config={"displayModeBar": False})
        st.caption("Lines: the forest's average at each jamming level, other settings as chosen. Shaded: where 80% "
                   "of its 200 trees land. Try changing the guidance link or jamming resistance and watch the slope.")

        # ---------------- sensitivity
        _h(11, "Sensitivity", "which conditions move the effectiveness forecast")
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
                line=dict(color=CYAN, width=7), opacity=.9, marker=dict(size=10, color=[D.RED, D.GREEN]),
                hovertemplate=f"{html.escape(label)}<br>worst {html.escape(str(worst))}: {lo:.2f}"
                              f"<br>best {html.escape(str(best))}: {hi:.2f}<extra></extra>"))
        fig2.add_vline(x=m_new, line=dict(color=CYAN, dash="dot", width=1))
        fig2.update_xaxes(range=[1, 5], title="Predicted effectiveness: worst to best option")
        st.plotly_chart(_hud_fig(fig2, 330), width="stretch", config={"displayModeBar": False})
        st.caption("Each bar: one condition run through all its options, others held as set (red = worst, green = best).")

        # ---------------- hangar
        _h(12, "Hangar", "saved designs, ranked under the current conditions")
        with st.container(key="hud_hangar"):
            hangar = st.session_state.setdefault("lab_hangar", [])
            b1, b2, _ = st.columns([1.3, 1, 2])
            if b1.button("Add this design", icon=":material/add:", key="lab_hangar_add", type="primary",
                         width="stretch"):
                hangar[:] = [d for d in hangar if d["name"] != design["name"]][-(HANGAR_MAX - 1):] + [dict(design)]
            if b2.button("Clear", icon=":material/delete_sweep:", key="lab_hangar_clear", width="stretch",
                         disabled=not hangar):
                hangar.clear()
            if hangar:
                high = {**cond, "ew_level": "High"}
                scored = []
                for d in hangar:
                    r, a, dw = _design_rows(d, cond)
                    now = float(_forecast(model, [r], [a] if a else None, dw)[0][0])
                    surv = float(_predict(model, "p_rating_survivability", [r], [a] if a else None, dw)[0])
                    loss = float(_predict(model, "lost", [r], [a] if a else None, dw)[0])
                    r2, a2, _ = _design_rows(d, high)
                    jam = float(_forecast(model, [r2], [a2] if a2 else None, dw)[0][0])
                    scored.append((now, surv, loss, jam, d))
                scored.sort(key=lambda x: -x[0])

                def tags(d):
                    ch = [f"{P.ATTRS[k][0].split(' /')[0].split(' (')[0]} {P.label(k, d['profile'][k])}"
                          for k in P.ORDER if d["profile"][k] != P.profile(d["base"], d["base_cat"], D.CATALOG)[0][k]]
                    ch += [d["origin"]] + (["field-mod"] if d["field_mod"] else []) + (
                        [f"{int(d['w'] * 100)}% {d['alt_cat']}"] if d["alt_cat"] else [])
                    return " · ".join(html.escape(x) for x in [d["base"]] + ch)
                rows_html = "".join(
                    f"<tr><td class='rk'>#{i}</td><td><b>{html.escape(d['name'])}</b><br><span class='sm'>{tags(d)}"
                    f"</span></td><td style='width:24%'>{_bar(now, _eff_color(now))}</td>"
                    f"<td class='n' style='color:{_eff_color(now)}'>{now:.1f}</td>"
                    f"<td class='n' style='color:{_eff_color(surv)}'>{surv:.1f}</td>"
                    f"<td class='n' style='color:{_outcome_color('lost', loss)}'>{loss:.0%}</td>"
                    f"<td class='n' style='color:{_eff_color(jam)}'>{jam:.1f}</td></tr>"
                    for i, (now, surv, loss, jam, d) in enumerate(scored, 1))
                st.html(f"<table class='hangar'><tr><th></th><th>Design</th><th></th><th>Effect.</th>"
                        f"<th>Survival</th><th>Loss</th><th>Heavy EW</th></tr>{rows_html}</table>")
                st.caption("Scores update as you move the conditions. 'Heavy EW' = effectiveness with enemy jamming "
                           "set to High.")
            else:
                st.caption("Save a few variants (more armor, a fiber-optic link, a different class blend) to compare "
                           "them side by side.")

        # ---------------- trust
        _h(13, "Can you trust it?")
        with st.container(key="hud_trust"):
            sc = model["scores"]
            body = "".join(
                f"<tr><td>{OUTCOMES[t][0]}</td><td class='n'>{s['model']:.3f}</td><td class='n'>{s['baseline']:.3f}</td>"
                f"<td class='n' style='color:{D.GREEN if s['model'] < s['baseline'] else D.RED}'>"
                f"{1 - s['model'] / s['baseline']:+.0%}</td><td class='sm'>{s['metric']}</td></tr>"
                for t, s in sc.items() if s["baseline"])
            st.html(f"<table class='hangar'><tr><th>Outcome model</th><th>Model</th><th>Baseline</th>"
                    f"<th>Better by</th><th>Error</th></tr>{body}</table>")
            st.caption(f"Tested on {model['n_test']} platform cards from {model['reports_test']} reports none of the "
                       "models saw (split by report, seed 42). Baseline = always guessing the average (loss: the "
                       "base rate). Lower error is better. Effectiveness, three models: "
                       + ", ".join(f"{k.lower()} {v:.2f}" for k, v in model["mae"].items()) + ".")
            if synthetic:
                # does the model recover what the generator planted?
                ref = {**ref_row}
                armor = [float(_predict(model, "p_rating_survivability", [{**ref, "protection": lv}])[0]) for lv in (0, 3)]
                jam = {lk: float(_forecast(model, [{**ref, "category": "FPV / Small Strike UAS", "mobility": "Airborne",
                                                    "link": lk, "ew_hardening": 0, "ew_level": "High"}])[0][0])
                       for lk in ("Radio data link", "Wire / fiber-optic")}
                checks = [("More armor raises predicted survival", armor[1] - armor[0] > 0.3,
                           f"{armor[0]:.2f} → {armor[1]:.2f}"),
                          ("Under heavy jamming a fiber-optic link beats a radio link",
                           jam["Wire / fiber-optic"] - jam["Radio data link"] > 0.3,
                           f"{jam['Radio data link']:.2f} vs {jam['Wire / fiber-optic']:.2f}")]
                st.html("<div class='mods'>" + "".join(
                    f"<div class='mod'><span class='pts' style='color:{D.GREEN if ok else D.ORANGE}'>"
                    f"{'PASS' if ok else 'NOT FOUND'}</span><span class='what'>Planted pattern: {html.escape(lbl)} "
                    f"<span class='sm'>({val})</span></span></div>" for lbl, ok, val in checks) + "</div>")
                st.caption("The sample generator planted these effects on purpose; a PASS shows the models can find a "
                           "real pattern. It is a test of the method, not evidence about any weapon. NOT FOUND usually "
                           "means the reports are too few or were generated before the effect was planted.")
            n_alt = int((cards["category"] == alt_cat).sum()) if has_alt else None
            thin = n_cat < THIN_CLASS or (has_alt and n_alt < THIN_CLASS)
            st.markdown(
                f"- **{n_cat}** reports for {base_cat}" + (f", **{n_alt}** for {alt_cat}" if has_alt else "")
                + (". :orange[Thin data: treat this class's forecast as a rough sketch.]" if thin else ".") + "\n"
                "- Characteristics are coarse estimates, and the reports never record them directly, so the models "
                "learn them only through which platforms were used. Effects are associations in the reports, not "
                "proven causes (heavily armored systems may be sent on harder missions).\n"
                "- The models cannot see anything outside the seven characteristics (exact range, seeker type, "
                "speed, cost).")
            if sc.get(TARGET, {}).get("baseline") and sc[TARGET]["model"] >= sc[TARGET]["baseline"]:
                st.warning("On this data the effectiveness forest does no better than guessing the average, so the "
                           "forecasts above are not informative yet. More reports are needed.", icon=":material/warning:")
    return True
