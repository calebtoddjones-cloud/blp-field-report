"""Build analyst_base.html for the web app's Insights page from the desktop project's html/template.html.

Same page as field_report_v3.html (all analyst features: Insights with Tactical / Officer / Sector map, Scorecard with
radar, heatmap, weapon profile and 3D viewer), with these LIVE-mode patches:
  * data comes from window.__BLP_DB__ (injected by the web app from field_reports.db) instead of browser storage
  * read-only: nothing is saved in the browser; load/remove sample, delete and restore are hidden
  * tabs: INSIGHTS, SCORECARD, REPORT LIST (the list is needed for "Open these reports" on the map); no form tab
Run:  python build_analyst.py <path to FieldReport/html>  -> writes analyst_base.html next to this script.
"""
import subprocess, sys, pathlib, shutil
HERE = pathlib.Path(__file__).resolve().parent
src_html = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
t = (src_html / "template.html").read_text(encoding="utf-8")

def sub(old, new, count=1):
    global t
    assert t.count(old) == count, (old[:60], t.count(old))
    t = t.replace(old, new)

sub('<script>\n"use strict";', '<script>/*__BLP_DB__*/</script>\n<script>\n"use strict";')
sub('let DB=loadDB();', 'const LIVE=!!window.__BLP_DB__;\nlet DB=LIVE?normalizeDB(window.__BLP_DB__):loadDB();\nif(LIVE)document.body.classList.add("live");')
sub('function saveDB(){try{', 'function saveDB(){if(LIVE)return;try{')
sub('const TABS=[["form","NEW REPORT"],["db","REPORT DATABASE"],["ins","INSIGHTS"],["score","SCORECARD"]];',
    'const TABS=(window.__BLP_DB__)?[["ins","INSIGHTS"],["score","SCORECARD"],["db","REPORT LIST"]]:[["form","NEW REPORT"],["db","REPORT DATABASE"],["ins","INSIGHTS"],["score","SCORECARD"]];')
sub('selectTab(TABS[(i+1)%4][0],true);', 'selectTab(TABS[(i+1)%TABS.length][0],true);')
sub('selectTab(TABS[(i+3)%4][0],true);', 'selectTab(TABS[(i+TABS.length-1)%TABS.length][0],true);')
sub('  $("#status").textContent=`REPORTS ${shownCount}/${total}   |   `+(storageOK?',
    '  if(LIVE){$("#status").textContent=`REPORTS ${shownCount}/${total}   |   LIVE DATA from the web app database (read-only view). Add reports on the New report page.`;return;}\n  $("#status").textContent=`REPORTS ${shownCount}/${total}   |   `+(storageOK?')
sub('  if(DB.reports.length===0&&!flagged){generateSampleData(SEED_COUNT);seeded=true;}',
    '  if(!LIVE&&DB.reports.length===0&&!flagged){generateSampleData(SEED_COUNT);seeded=true;}')
sub('renderDB();selectTab("form");', 'renderDB();if(LIVE){$("#tab-form").hidden=true;selectTab("ins");}else selectTab("form");')
sub('</style>\n</head>', 'body.live .topbar{display:none}\nbody.live #loadSamples,body.live #rmSamples,body.live #delSel,body.live #resJson{display:none}\n</style>\n</head>')
work = HERE / "_build"
if work.exists(): shutil.rmtree(work)
(work / "html").mkdir(parents=True)
(work / "html" / "template.html").write_text(t, encoding="utf-8")
shutil.copy(src_html / "build.py", work / "html" / "build.py")
for name in ("platform_info.json", "platform_specs.json"):
    shutil.copy(src_html.parent / name, work / name)
shutil.copytree(src_html.parent / "assets", work / "assets")
subprocess.run([sys.executable, str(work / "html" / "build.py"), str(HERE / "analyst_base.html")], check=True)
