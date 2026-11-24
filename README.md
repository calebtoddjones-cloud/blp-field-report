# BLP Field Reporting: web app (Streamlit prototype)

Web version of the **level 1 (Tactical)** report from the desktop app
`FieldReport/field_report_app_v3.py`: same questions, same stored values, same database
layout and upgrade steps, so one `field_reports.db` works in both. Levels 2–4 are not
asked here yet; level 2–4 rows already in a database are left alone and kept out of the
tables and charts (as the desktop app does for its Scorecard and Insights).

**Everything here is illustrative.** The platform catalog is not authoritative, and the
"Load sample reports" button makes synthetic data with a built-in EW pattern. 

## Run it on your computer

```
cd "C:\Users\caleb\I4D workspace\FieldReportWeb"
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at http://localhost:8501. The database file `field_reports.db` is
created in this folder the first time you submit or load sample data.

To use the desktop app's database instead (PowerShell):

```
$env:BLP_DB = "C:\Users\caleb\I4D workspace\FieldReport\field_reports.db"
streamlit run app.py
```

An older (v2) database is upgraded the same way the desktop v3 app does it, with the same
one-time backup copy saved next to it first.

## What's in it

- **New report**: 5 steps (Unit & task → Conditions → Platforms & ratings →
  Malfunctions & result → Review & submit), or Quick mode with only the required answers.
  Each platform gets its own card: how it performed, effectiveness (required) and the six
  traits (maintenance, spare parts, platform survivability, crew survivability, lethality,
  how hard to hit). Conditions and the adversary section are optional, as in v3; the
  adversary records nothing unless switched on. The limiting-factor question appears only
  when the outcome isn't Success, and impact only when a malfunction is picked.
  Answers are colored red (worst) to green (best); neutral questions are blue and
  "Unknown" is gray. Missing required answers show in red, with Fix buttons on Review.
- **Report database**: filters, a color-coded table, and a detail panel with each
  platform's own ratings; CSV export, sample data, delete.
- **Insights**: the analyst console from the desktop project's v3 HTML page (`analyst_base.html`,
  built by `build_analyst.py`), reading this app's live database read-only: Tactical tables,
  Officer Insights (levels 2-4), Sector map, Scorecard (adjusted averages, composite, radar,
  heatmap), weapon profiles and the 3D viewer. A "Quick summary" tab keeps the simpler charts.
- **Graphs**: interactive views that redraw from the reports every time: platform networks
  (used together / platform to malfunction / category hubs), learning curves (running averages
  settling as reports arrive), 3D effectiveness landscape and trait space, a flow (Sankey) from
  conditions to outcome to limiting factor, and animated bubbles week by week. Plotly.
- **Lab**: design a new system and forecast how it would do. Start from the closest existing
  system (e.g. HAWK); its estimated capability profile (armor, signature, mobility, guidance / control
  link, jamming resistance, reach, crew; `profiles.py`, illustrative estimates) sets the design sliders.
  Also: origin, field-modified, and "behaves partly like another class" (an analyst assumption that blends
  two classes' forecasts). Separate models forecast effectiveness, platform survival, crew survival, ease of
  maintenance, how hard it is to hit and loss risk (all from questions the form already asks), so changes
  show up as trade-offs. Panels: 3D hologram of the base system, status bar, trade-off profile (original → design
  arrows), stat modifiers (waterfall), sweet spot finder (mission-priority weights; sweep one characteristic or
  a two-way heatmap, with original, design and best marked), trade space (every reported system under the chosen conditions), nearest real systems with an
  extrapolation alert, three effectiveness models side by side, EW stress test, sensitivity, a session
  "hangar" of saved designs, and a trust panel (holdout error vs baseline per outcome, split by report,
  seed 42, plus a planted-pattern check on synthetic data). Needs at least 60 rated platform cards.

  The report form is unchanged: characteristics are looked up by platform name at analysis time. The web
  app's sample generator now plants profile-driven effects (link and jamming resistance set the EW penalty;
  armor and signature drive survival with diminishing returns; armor makes maintenance harder, faster at
  the heavy end; planted sweet spots: armor helps effectiveness up to light/medium then hurts at heavy, reach
  peaks at medium), so its sample data no longer matches the desktop app's generator.

## Files

| File | What it does |
|---|---|
| `app.py` | The three pages and the app shell |
| `data.py` | Catalog, answer options and their colors, database (ported from v3), sample-data generator (v3's, level 1 only) |
| `graphs.py` | The Graphs page |
| `lab.py` | The Lab page: design a system and forecast six outcomes (scikit-learn models, three.js hologram) |
| `profiles.py` | Estimated capability profile for every catalog platform (illustrative, replace with sourced values) |
| `analyst_base.html` | Analyst console page with the 3D models baked in (rebuild with `python build_analyst.py "..\\FieldReport\\html"`) |
| `widgets.py` | Color-coded answer buttons, colored dropdowns, badges, page styling |
| `.streamlit/config.toml` | Dark theme |
| `requirements.txt` | Python packages (Streamlit 1.67 or newer) |

To change an answer's color or wording, edit its option list in `data.py`, e.g.
`EW_LEVELS = [("None", GREEN), ("Low", LIME), ("Medium", ORANGE), ("High", RED)]`.
If the team changes a question in the desktop app, change it here too so the two stay
in step.

## Put it online (free, Streamlit Community Cloud)

1. **GitHub.** Sign in at https://github.com (make an account if needed). Click **+** (top
   right) > **New repository**, name it e.g. `blp-field-report`, leave everything else as is,
   click **Create repository**. On the next page click **uploading an existing file**, drag in
   everything from this folder *including the `.streamlit` folder* (but not
   `field_reports.db`), and click **Commit changes**.
2. **Streamlit.** Go to https://share.streamlit.io, sign in with GitHub, click
   **Create app** > deploy from GitHub. Pick the repository, branch `main`, main file
   `app.py`. Choose a custom address if you like (e.g. `blp-field-report`).
3. **Secrets** (same page: **Advanced settings**, or later under the app's **Settings >
   Secrets**). Paste, with your own passcode:
   ```
   BLP_ADMIN_CODE = "pick-a-passcode"
   BLP_AUTO_SAMPLE = "1"
   ```
   The passcode limits deleting and loading data to you (Report database > Manage data).
   Auto sample refills 150 synthetic reports whenever the app starts empty.
4. Click **Deploy**. When it's up, click **Share** (top right of the app) to show a big QR
   code of its address for people to scan.

Caveats of the free hosting: the app sleeps after a period with no visitors (the first
visit wakes it in under a minute, so open it yourself before class), and the database file is
not permanent, so reports submitted there can disappear when the app restarts. Export the CSV
if you want to keep them. Anyone with the link can open the app and submit reports.

To change the app later, upload the changed file to the same GitHub repository; Streamlit
picks it up within a minute.
