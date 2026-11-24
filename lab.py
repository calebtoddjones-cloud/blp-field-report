"""Lab page: sketch a new system from one the model already knows, set the conditions, see a forecast.

How it works
- Three models are trained on the level 1 platform cards in the live database, predicting effectiveness
  (1-5) from platform category, origin, field-modified, and the battlefield conditions on the report:
    decision tree (one readable set of yes/no questions), random forest (200 trees, the main forecast),
    ridge regression with category x condition interactions (readable effects, alpha fixed, not tuned).
  After-the-fact observations (traits, malfunctions, outcome) are never inputs: a forecast has to work
  before the engagement, and in the sample data the traits are generated from effectiveness.
- Holdout split is by REPORT (seed 42); every model is compared with guessing the average. The models you
  play with are the ones that were tested. When the three disagree, the page says so.
- A "new system" starts from a known platform (its category and origin). Modifications the model can
  represent: a different origin, a field-modified build, and "behaves partly like another class", which
  blends the two classes' forecasts. The blend is an ANALYST ASSUMPTION, not something learned.
- 3D: the base platform's model (and a wireframe of the blended class) from the slimmed models already
  embedded in analyst_base.html, drawn with three.js from a CDN. The shape is the base system's;
  modifications are not drawn.

SYNTHETIC sample data has a planted EW pattern; results test the pipeline, they are not findings.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

import data as D
from graphs import FG, style

SEED = 42
MIN_CARDS = 60          # below this the page asks for more reports instead of training
THIN_CLASS = 20         # fewer cards than this in a class -> "thin data" warning
AGREE = 0.35            # three models within this many points of each other -> "agree"
RIDGE_ALPHA = 10.0      # fixed, not tuned
TARGET = "p_effectiveness"
CAT_FEATS = ["category", "origin", "ew_level", "ad_level", "time_of_day", "weather", "terrain",
             "unit_type", "task", "enemy_type", "enemy_strength", "enemy_adaptation"]
NUM_FEATS = ["p_field_modified"]
INTERACT = [c for c in CAT_FEATS if c not in ("category", "origin")]
COND_LABELS = {"ew_level": "Enemy EW / jamming", "ad_level": "Enemy air defense", "time_of_day": "Time of day",
               "weather": "Weather", "terrain": "Terrain", "unit_type": "Unit type", "task": "Primary task",
               "enemy_type": "Enemy type", "enemy_strength": "Enemy strength",
               "enemy_adaptation": "Enemy adaptation"}
LEVELS = [v for v in D.values(D.THREAT) if v != "Unknown"]          # None, Low, Medium, High
ORIGINS = ["U.S.", "Ukraine", "Other partner"]
MODEL_NAMES = ["Decision tree", "Random forest", "Ridge + interactions"]
MODEL_BLURB = {"Decision tree": "one set of yes/no questions",
               "Random forest": "200 trees averaged · main forecast",
               "Ridge + interactions": "straight-line effects per class"}

# HUD palette: cyan for the new system, steel for the base, amber for the blended class.
CYAN, STEEL, AMBER = "#3ee0ff", "#9aa7b4", "#e3b93c"
HUD_LINE, HUD_GRID, HUD_DIM = "rgba(62,224,255,.28)", "rgba(62,224,255,.10)", "#7f93a6"
MONO = "'JetBrains Mono', Consolas, monospace"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _prep(df):
    X = df.reindex(columns=CAT_FEATS + NUM_FEATS).copy()
    for c in CAT_FEATS:
        X[c] = X[c].fillna("Unknown").astype(str).replace({"": "Unknown"})
    for c in NUM_FEATS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X


def _prep_lin(df):
    """Same inputs plus 'category x condition' columns, so the ridge can learn class-specific effects."""
    X = _prep(df)
    for c in INTERACT:
        X[f"cat x {c}"] = X["category"] + " | " + X[c]
    return X


LIN_CATS = CAT_FEATS + [f"cat x {c}" for c in INTERACT]


def _pipe(est, cats):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                             ("num", "passthrough", NUM_FEATS)])
    return Pipeline([("pre", pre), ("model", est)])


@st.cache_resource(show_spinner="Training three models on the current reports…", max_entries=4)
def _train(_key, _cards):
    X, Xl, y, groups = _prep(_cards), _prep_lin(_cards), _cards[TARGET].astype(float), _cards["id"]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
    fits = {
        "Decision tree": _pipe(DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=SEED),
                               CAT_FEATS).fit(X.iloc[tr], y.iloc[tr]),
        "Random forest": _pipe(RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                                     random_state=SEED), CAT_FEATS).fit(X.iloc[tr], y.iloc[tr]),
        "Ridge + interactions": _pipe(Ridge(alpha=RIDGE_ALPHA), LIN_CATS).fit(Xl.iloc[tr], y.iloc[tr]),
    }
    mae = {"Guess the average": mean_absolute_error(
        y.iloc[te], DummyRegressor(strategy="mean").fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]))}
    for name, p in fits.items():
        Xt = Xl if name.startswith("Ridge") else X
        mae[name] = mean_absolute_error(y.iloc[te], np.clip(p.predict(Xt.iloc[te]), 1, 5))
    return {"fits": fits, "mae": {k: float(v) for k, v in mae.items()}, "n_train": int(len(tr)),
            "n_test": int(len(te)), "reports_test": int(groups.iloc[te].nunique())}


def _trees(model, rows):
    """Forest per-tree predictions, shape (n_trees, n_rows)."""
    pipe = model["fits"]["Random forest"]
    Xt = pipe["pre"].transform(_prep(pd.DataFrame(rows)))
    return np.stack([t.predict(Xt) for t in pipe["model"].estimators_])


def _forecast(model, rows, alt_rows=None, w=0.0):
    """Forest mean and 10-90% tree spread per row; optionally blended with a second class (weight w)."""
    t = _trees(model, rows)
    if alt_rows is not None and w > 0:
        t = (1 - w) * t + w * _trees(model, alt_rows)
    return t.mean(axis=0), np.percentile(t, 10, axis=0), np.percentile(t, 90, axis=0)


def _all_three(model, row, alt_row=None, w=0.0):
    """One prediction from each model for a single scenario (blended the same way)."""
    out = {}
    for name, pipe in model["fits"].items():
        prep = _prep_lin if name.startswith("Ridge") else _prep
        v = float(np.clip(pipe.predict(prep(pd.DataFrame([row])))[0], 1, 5))
        if alt_row is not None and w > 0:
            v = (1 - w) * v + w * float(np.clip(pipe.predict(prep(pd.DataFrame([alt_row])))[0], 1, 5))
        out[name] = v
    return out


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
.hangar .n {{ font-weight:800; font-size:1rem; text-align:right; width:64px; }}
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
    "Night + jamming": {"lab_ew": "High", "lab_ad": "Medium", "lab_tod": "Night", "lab_wx": "Overcast"},
    "Contested": {"lab_ew": "Medium", "lab_ad": "High", "lab_tod": "Dawn / dusk", "lab_wx": "Rain"},
    "Worst case": {"lab_ew": "High", "lab_ad": "High", "lab_tod": "Night", "lab_wx": "Fog"},
}
HANGAR_MAX = 8


def _apply_preset(name):
    st.session_state.update(PRESETS[name])


def _design_rows(d, cond):
    """(row, alt_row, w) for a saved or current design under the given conditions."""
    row = {"category": d["base_cat"], "origin": d["origin"], "p_field_modified": int(d["field_mod"]), **cond}
    alt = {**row, "category": d["alt_cat"]} if d["alt_cat"] else None
    return row, alt, d["w"]


def _status_tile(label, value, color, sub=""):
    return (f"<div class='tile'><div class='lbl'>{html.escape(label)}</div>"
            f"<div class='val' style='color:{color};text-shadow:0 0 16px {color}55'>{value}</div>"
            f"<div class='sub'>{sub}</div></div>")


def _mod_chip(label, pts):
    col = D.GREEN if pts > 0.05 else D.RED if pts < -0.05 else HUD_DIM
    sign = "▲" if pts > 0.05 else "▼" if pts < -0.05 else "■"
    return (f"<div class='mod'><span class='pts' style='color:{col}'>{sign} {pts:+.2f}</span>"
            f"<span class='what'>{html.escape(label)}</span></div>")


def render(conn):
    cards = D.platform_rows_df(conn)
    cards = cards[(cards["rated_individually"] == 1) & cards["p_effectiveness"].notna()].reset_index(drop=True)
    if len(cards) < MIN_CARDS:
        st.info(f"The Lab trains models on the reports, so it needs at least {MIN_CARDS} rated platform cards "
                f"(there are {len(cards)}). Add reports, or generate sample data above.", icon=":material/science:")
        return False

    st.html(f"<style>{HUD_CSS}</style>")
    key = hashlib.md5(cards[["id", TARGET] + CAT_FEATS + NUM_FEATS].to_csv(index=False).encode()).hexdigest()
    model = _train(key, cards)
    synthetic = bool((cards["notes"].fillna("") == "[sample data]").all())

    st.html("<div class='caveat'><span>⚠</span><span>A forecast from the reports, for exploring ideas. "
            + ("Everything here is <b>synthetic sample data with a planted EW pattern</b>: this shows the method "
               "working, it says nothing about real weapons. " if synthetic else
               "Some or all of these reports are synthetic sample data; treat the numbers as a method demo. ")
            + "The models only know a system's class, origin, whether it was field-modified, and the "
              "conditions below.</span></div>")

    counts = cards["platform"].value_counts()
    known = sorted(counts.index, key=lambda n: (D.PLATFORM_INFO.get(n, ("~",))[0], n))
    default = "HAWK" if "HAWK" in counts else counts.index[0]

    hero = st.container()          # filled after the controls below are read
    status = st.container()
    left, right = st.columns([5, 7], gap="large")

    # ------------------------------------------------------------- controls
    with left:
        _h(1, "Base platform", "closest system the models know")
        with st.container(key="hud_base"):
            base = st.selectbox("Closest existing system", known, index=known.index(default), key="lab_base",
                                format_func=lambda n: f"{n}  ·  {counts[n]} report{'s' if counts[n] != 1 else ''}")
            row0 = cards[cards["platform"] == base].iloc[0]
            base_cat, base_origin = row0["category"], row0["origin"]
            n_cat = int((cards["category"] == base_cat).sum())
            st.caption(f"Class **{base_cat}** · {n_cat} reports in this class · origin **{base_origin}**")
            name = st.text_input("Designation for the new system", value=f"{base}-X", key=f"lab_name_{base}")

        _h(2, "Modifications", "what the models can represent")
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
                st.caption("Analyst assumption: the forecast blends the two classes' learned behaviour in this "
                           "ratio. The models have not seen this hybrid.")

        _h(3, "Mission conditions")
        with st.container(key="hud_cond"):
            for k, v in PRESETS["Quiet day"].items():
                st.session_state.setdefault(k, "Low" if k == "lab_ad" else v)
            st.caption("Mission presets")
            pc = st.columns(len(PRESETS))
            for c, (pname, _) in zip(pc, PRESETS.items()):
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

    # ------------------------------------------------------------- forecast
    has_alt = alt_cat != "None"
    design = {"name": name or "New system", "base": base, "base_cat": base_cat, "origin": origin,
              "field_mod": bool(field_mod), "alt_cat": alt_cat if has_alt else None, "w": w}
    new_row, alt_row, _ = _design_rows(design, cond)
    ref_row = {"category": base_cat, "origin": base_origin, "p_field_modified": 0, **cond}
    blend = (lambda rows: [{**r, "category": alt_cat} for r in rows]) if has_alt else (lambda rows: None)

    m_new, lo_new, hi_new = (float(a[0]) for a in _forecast(model, [new_row], blend([new_row]), w))
    m_ref, lo_ref, hi_ref = (float(a[0]) for a in _forecast(model, [ref_row]))
    delta = round(m_new - m_ref, 1) + 0.0     # + 0.0 turns -0.0 into 0.0
    three = _all_three(model, new_row, alt_row, w)
    spread = max(three.values()) - min(three.values())
    agree = spread < AGREE

    # ------------------------------------------------------------- hero + status bar
    with hero:
        mods = []
        if origin != base_origin:
            mods.append(("Origin", origin))
        if field_mod:
            mods.append(("Build", "field-modified"))
        if has_alt:
            mods.append(("Behaves like", alt_cat))
        if not _hologram(name, base, base_cat, n_cat, mods, alt_cat if has_alt else None):
            st.caption("No 3D model for this system.")
    with status, st.container(key="hud_status"):
        t = st.columns(4)
        t[0].html(_status_tile(f"{(name or 'New system')[:28]} · forecast", f"{m_new:.1f}", _eff_color(m_new),
                               f"80% of trees {lo_new:.1f}–{hi_new:.1f}"))
        t[1].html(_status_tile(f"{base} as fielded", f"{m_ref:.1f}", STEEL, "same conditions"))
        t[2].html(_status_tile("Change", f"{'▲' if delta > 0 else '▼' if delta < 0 else '■'} {delta:+.1f}",
                               D.GREEN if delta > 0 else D.RED if delta < 0 else HUD_DIM, "new vs base, points"))
        t[3].html(_status_tile("Three models", "AGREE" if agree else "SPLIT", D.GREEN if agree else D.ORANGE,
                               f"spread {spread:.2f} pts"))
        st.html(f"<div class='cond-line'>CONDITIONS · EW {html.escape(cond['ew_level'])} · AIR DEFENSE "
                f"{html.escape(cond['ad_level'])} · {html.escape(cond['time_of_day'])} · {html.escape(cond['weather'])}"
                f" · {html.escape(cond['terrain'])}</div>")

    with right:
        _h(4, "Stat modifiers", "what each change does, one at a time")
        with st.container(key="hud_modifiers"):
            steps, row, prev = [], dict(ref_row), m_ref
            if origin != base_origin:
                row = {**row, "origin": origin}
                v = float(_forecast(model, [row])[0][0]); steps.append((f"Origin: {origin}", v - prev)); prev = v
            if field_mod:
                row = {**row, "p_field_modified": 1}
                v = float(_forecast(model, [row])[0][0]); steps.append(("Field-modified build", v - prev)); prev = v
            if has_alt and w > 0:
                steps.append((f"Behaves {int(w * 100)}% like {alt_cat}", m_new - prev))
            chips = "".join(_mod_chip(lbl, pts) for lbl, pts in steps) or \
                f"<div class='mod'><span class='what'>Stock build: no modifications set (section 02).</span></div>"
            st.html(f"<div class='mods'><div class='mod'><span class='pts' style='color:{STEEL}'>{m_ref:.2f}</span>"
                    f"<span class='what'>{html.escape(base)} as fielded</span></div>{chips}"
                    f"<div class='mod tot'><span class='pts' style='color:{_eff_color(m_new)}'>{m_new:.2f}</span>"
                    f"<span class='what'>{html.escape(name or 'New system')}</span></div></div>")
            st.caption("Each line adds one modification to the one above and shows how much the forest's forecast "
                       "moves. The order matters a little, because effects can interact.")

        _h(5, "Three models", "independent forecasts for the new system")
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

        _h(6, "Stress test", "enemy jamming from none to high")
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
                   "of its 200 trees land. Dotted line: the level you picked.")

        _h(7, "Sensitivity", "which conditions move this forecast")
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
        st.caption("Each bar: one condition run through all its options, others held as set. Long bars are what "
                   "this system is sensitive to (red = worst option, green = best).")

        _h(8, "Hangar", "saved designs, ranked under the current conditions")
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
                    r2, a2, _ = _design_rows(d, high)
                    jam = float(_forecast(model, [r2], [a2] if a2 else None, dw)[0][0])
                    scored.append((now, jam, d))
                scored.sort(key=lambda x: -x[0])
                rows_html = "".join(
                    f"<tr><td class='rk'>#{i}</td><td><b>{html.escape(d['name'])}</b><br><span class='sm'>"
                    f"{html.escape(d['base'])} · {html.escape(d['origin'])}"
                    f"{' · field-mod' if d['field_mod'] else ''}"
                    f"{' · ' + str(int(d['w'] * 100)) + '% ' + html.escape(d['alt_cat']) if d['alt_cat'] else ''}"
                    f"</span></td><td style='width:34%'>{_bar(now, _eff_color(now))}</td>"
                    f"<td class='n' style='color:{_eff_color(now)}'>{now:.1f}</td>"
                    f"<td class='n' style='color:{_eff_color(jam)}'>{jam:.1f}</td></tr>"
                    for i, (now, jam, d) in enumerate(scored, 1))
                st.html(f"<table class='hangar'><tr><th></th><th>Design</th><th></th><th>Now</th>"
                        f"<th>Heavy EW</th></tr>{rows_html}</table>")
                st.caption("Scores update as you move the conditions. 'Heavy EW' is the same design with enemy "
                           "jamming set to High.")
            else:
                st.caption("Save a few variants (different origins, blends, builds) to compare them side by side.")

        _h(9, "Can you trust it?")
        mae = model["mae"]
        base_mae = mae["Guess the average"]
        gain = 1 - mae["Random forest"] / base_mae if base_mae else 0
        n_alt = int((cards["category"] == alt_cat).sum()) if has_alt else None
        thin = n_cat < THIN_CLASS or (has_alt and n_alt < THIN_CLASS)
        st.markdown(
            f"- Trained on **{model['n_train']}** platform cards; tested on **{model['n_test']}** cards from "
            f"**{model['reports_test']}** reports none of the models saw. Average miss: forest "
            f"**{mae['Random forest']:.2f}**, tree **{mae['Decision tree']:.2f}**, ridge "
            f"**{mae['Ridge + interactions']:.2f}**, vs **{base_mae:.2f}** for always guessing the average "
            f"(forest **{gain:.0%} {'smaller' if gain >= 0 else 'larger'}**).\n"
            f"- **{n_cat}** reports for {base_cat}" + (f", **{n_alt}** for {alt_cat}" if has_alt else "")
            + (". :orange[Thin data: treat this class's forecast as a rough sketch.]" if thin else ".") + "\n"
            "- The models cannot see range, seeker, speed or any design detail. A modification that changes "
            "those only shows up once units report on the new system (or a new question captures it).")
        if gain <= 0:
            st.warning("On this data the forest does no better than guessing the average, so the forecasts "
                       "above are not informative yet. More reports are needed.", icon=":material/warning:")
    return True
