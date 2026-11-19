"""Data layer for the BLP field report web app.

Mirrors the desktop app FieldReport/field_report_app_v3.py (level 1, "Tactical"):
same questions, same SQLite schema, same upgrade steps for older databases, so one
field_reports.db works in both. Levels 2-4 are not asked here; any level 2-4 rows a
v3 database already holds are left alone and kept out of every table and chart.

NOTE: the platform catalog is ILLUSTRATIVE (brainstorming data, not accurate).
Sample data is SYNTHETIC with a planted EW-sensitivity pattern; it tests the app, it is
not evidence.
"""

import os
import random
import shutil
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Set BLP_DB to a path to share a database file with the desktop app.
DB_PATH = Path(os.environ.get("BLP_DB") or Path(__file__).parent / "field_reports.db")

# ---------------------------------------------------------------------------
# Colors: a 5-step scale from worst (red) to best (green), plus neutrals.
# ---------------------------------------------------------------------------
RED, ORANGE, YELLOW, LIME, GREEN = "#e05d5d", "#ec8a3c", "#e3b93c", "#a3c94a", "#46b673"
GRAY, BLUE = "#8d99a6", "#4a9bd1"
SCALE_5 = [RED, ORANGE, YELLOW, LIME, GREEN]  # index 0 = worst, 4 = best

# ---------------------------------------------------------------------------
# Platform catalog (illustrative). category -> [(name, origin)]
# ---------------------------------------------------------------------------
CATALOG = {
    "Loitering / Strike UAS": [
        ("Switchblade 300", "U.S."), ("Switchblade 600", "U.S."), ("Phoenix Ghost", "U.S."),
        ("Hero-120", "Other partner"), ("Warmate", "Other partner"),
        ("Liutyi long-range UAS", "Ukraine"), ("RAM II strike UAS", "Ukraine"),
    ],
    "FPV / Small Strike UAS": [
        ("FPV 7-inch", "Ukraine"), ("FPV 10-inch heavy", "Ukraine"),
        ("Fiber-optic FPV", "Ukraine"), ("Drop-munition quadcopter", "Ukraine"),
        ("Commercial quad (modified)", "Other partner"),
    ],
    "Recon / ISR UAS": [
        ("RQ-20 Puma", "U.S."), ("RQ-11 Raven", "U.S."), ("ScanEagle", "U.S."),
        ("Leleka-100", "Ukraine"), ("Furia", "Ukraine"), ("Shark", "Ukraine"),
        ("Bayraktar TB2", "Other partner"), ("Commercial recon quad", "Other partner"),
    ],
    "Counter-UAS": [
        ("Coyote Block 2", "U.S."), ("Gepard SPAAG", "Other partner"),
        ("Interceptor drone", "Ukraine"), ("Acoustic sensor network", "Ukraine"),
        ("Mobile fire group (MG)", "Ukraine"),
    ],
    "Air Defense": [
        ("Patriot PAC-3", "U.S."), ("NASAMS", "Other partner"), ("IRIS-T SLM", "Other partner"),
        ("Stinger (MANPADS)", "U.S."), ("Avenger", "U.S."), ("HAWK", "U.S."),
        ("Buk-M1", "Ukraine"), ("S-300", "Ukraine"),
    ],
    "Anti-Armor": [
        ("Javelin", "U.S."), ("TOW", "U.S."), ("AT4", "Other partner"),
        ("NLAW", "Other partner"), ("Stugna-P", "Ukraine"), ("Skif", "Ukraine"),
    ],
    "Artillery / Mortars": [
        ("M777 155mm", "U.S."), ("M109 Paladin", "U.S."), ("M252 81mm mortar", "U.S."),
        ("PzH 2000", "Other partner"), ("CAESAR", "Other partner"), ("Krab", "Other partner"),
        ("Bohdana 2S22", "Ukraine"), ("2S1 Gvozdika", "Ukraine"), ("120mm mortar", "Ukraine"),
    ],
    "Rockets / Missiles": [
        ("HIMARS (GMLRS)", "U.S."), ("M270 MLRS", "U.S."), ("ATACMS", "U.S."),
        ("Storm Shadow", "Other partner"), ("Neptune", "Ukraine"),
        ("FP-5 Flamingo", "Ukraine"), ("BM-21 Grad", "Ukraine"),
    ],
    "Precision Munitions": [
        ("Excalibur 155mm", "U.S."), ("JDAM-ER", "U.S."), ("GLSDB", "U.S."),
    ],
    "Armored / Vehicles": [
        ("M1A1 Abrams", "U.S."), ("M2 Bradley", "U.S."), ("Stryker", "U.S."),
        ("M113", "U.S."), ("MaxxPro MRAP", "U.S."), ("HMMWV", "U.S."),
        ("Leopard 2A6", "Other partner"), ("Challenger 2", "Other partner"),
        ("T-64BV", "Ukraine"), ("BMP-2", "Ukraine"),
    ],
    "Small Arms / Crew-Served": [
        ("M4 carbine", "U.S."), ("M249 SAW", "U.S."), ("M240B", "U.S."),
        ("M2 .50 cal", "U.S."), ("M107 .50 sniper", "U.S."), ("AK-74", "Ukraine"),
    ],
    "EW / C2 / Comms": [
        ("Starlink terminal", "Other partner"), ("Delta situational awareness", "Ukraine"),
        ("Kvertus EW system", "Ukraine"), ("Bukovel-AD", "Ukraine"),
        ("Silent Archer EW", "Other partner"), ("AN/PRC-152 radio", "U.S."),
        ("AN/PRC-117G radio", "U.S."),
    ],
    "Naval / USV": [
        ("Magura V5", "Ukraine"), ("Sea Baby", "Ukraine"), ("Harpoon (coastal)", "Other partner"),
    ],
}
PLATFORM_INFO = {n: (c, o) for c, items in CATALOG.items() for n, o in items}

# How sensitive each category is to enemy EW (used only to pattern the sample data).
EW_SENSITIVITY = {
    "Loitering / Strike UAS": 0.8, "FPV / Small Strike UAS": 1.0, "Recon / ISR UAS": 0.9,
    "Counter-UAS": 0.4, "EW / C2 / Comms": 0.6, "Precision Munitions": 0.5,
    "Rockets / Missiles": 0.3,
}

# ---------------------------------------------------------------------------
# Questionnaire options (wording and stored values copied from field_report_app_v3.py).
# Plain lists = neutral dropdowns. Lists of (value, color[, label]) = colored answers,
# in screen order. Colors run red (worst) to green (best); blue = neutral; gray = unknown.
# ---------------------------------------------------------------------------
UNIT_TYPES = ["Drone crew", "Outpost", "Infantry squad", "Artillery section",
              "EW team", "Mechanized platoon", "Other"]
TASKS = ["Reconnaissance", "Strike", "Defense", "Counter-drone",
         "Resupply / logistics", "EW / jamming", "Other"]
LOCATIONS = ["Kharkiv", "Kupiansk", "Lyman / Siversk", "Kramatorsk / Sloviansk",
             "Chasiv Yar / Toretsk", "Pokrovsk", "Kurakhove / Vuhledar", "Huliaipole",
             "Orikhiv / Zaporizhzhia", "Kherson / Dnipro River", "Black Sea / Odesa",
             "Northern border (Sumy)", "Rear area / deep strike", "Other"]
ENEMY_TYPES = ["Unknown", "Infantry", "Mechanized / armor", "Artillery", "UAS operators",
               "EW unit", "Mixed"]

WEATHER = [("Clear", GREEN), ("Overcast", LIME), ("Rain", YELLOW), ("Fog", ORANGE),
           ("Snow", ORANGE), ("High wind", ORANGE)]
TIME_OF_DAY = [("Day", BLUE), ("Night", BLUE), ("Dawn / dusk", BLUE)]
TERRAIN = [("Open field", BLUE), ("Forest / treeline", BLUE), ("Urban", BLUE),
           ("Trench line", BLUE), ("River / wetland", BLUE)]
# Enemy EW sits with the conditions (optional since 2026-11-13). "None" is a real answer; blank = not given.
EW_LEVELS = [("None", GREEN), ("Low", LIME), ("Medium", ORANGE), ("High", RED)]
THREAT = [("Unknown", GRAY), ("None", GREEN), ("Low", LIME), ("Medium", ORANGE), ("High", RED)]
ENEMY_STRENGTH = [("Unknown", GRAY), ("1-5", GREEN), ("6-20", LIME), ("21-50", YELLOW),
                  ("51-100", ORANGE), ("100+", RED)]
ENEMY_ADAPT = [("Unknown", GRAY), ("None observed", GREEN), ("Tactic change", ORANGE),
               ("Countermeasure to platform", RED)]
ADVERSARY_KEYS = ["enemy_type", "enemy_strength", "ad_level", "enemy_adaptation"]

# Mission outcome: same four stored values as v2/v3. Definitions are v3's DRAFT wording.
OUTCOMES = [("Success", GREEN), ("Partial success", YELLOW, "Partial"),
            ("Aborted", ORANGE), ("Failure", RED)]
OUTCOME_DEFS = {
    "Success": "Objective fully achieved.",
    "Partial success": "Part of the objective achieved.",
    "Failure": "Objective not achieved.",
    "Aborted": "Mission stopped early or not attempted.",
}
# Asked only when the outcome is not Success.
OUTCOME_DRIVERS = [(d, BLUE) for d in [
    "Equipment performance / malfunction", "Enemy EW / countermeasures",
    "Enemy fires / attrition", "Supply / sustainment shortage", "Weather / terrain",
    "Training / procedures", "Intelligence / targeting error", "Other"]]

# Per-platform card.
STATUS_LOST = "Lost / destroyed"
PLATFORM_STATUS = [("Performed as expected", GREEN), ("Degraded but usable", YELLOW),
                   ("Failed to perform", ORANGE), (STATUS_LOST, RED)]
RATING = [(1, RED, "1"), (2, ORANGE, "2"), (3, YELLOW, "3"), (4, LIME, "4"), (5, GREEN, "5")]
EFFECTIVENESS = [(1, RED, "1 · Very poor"), (2, ORANGE, "2 · Poor"), (3, YELLOW, "3 · Mixed"),
                 (4, LIME, "4 · Good"), (5, GREEN, "5 · Excellent")]
# (key, label, hint, required): effectiveness, then the six areas the sponsor stressed, in order.
PLATFORM_RATINGS = [
    ("effectiveness", "Effectiveness (overall)", "5 = best", True),
    ("rating_ease", "Ease of maintenance", "5 = easy", False),
    ("rating_sustainment", "Spare parts availability", "5 = readily available", False),
    ("rating_survivability", "Platform survivability", "5 = survives well", False),
    ("rating_crew_survivability", "Crew survivability", "5 = crew survives well", False),
    ("rating_lethality", "Lethality", "5 = very lethal", False),
    ("rating_hard_to_hit", "How hard is it to hit", "5 = very hard to hit", False),
]
TRAITS = PLATFORM_RATINGS[1:]
TRAIT_SHORT = {"rating_ease": "Maintenance", "rating_sustainment": "Spare parts",
               "rating_survivability": "Platform survival", "rating_crew_survivability": "Crew survival",
               "rating_lethality": "Lethality", "rating_hard_to_hit": "Hard to hit"}
YES_NO_NEUTRAL = [(0, BLUE, "No"), (1, BLUE, "Yes")]

MAL_NONE = "None observed"
MAL_TYPES = ["Jam / stoppage", "Misfire / dud", "Comms / data-link loss",
             "EW / GPS interference", "Guidance / targeting failure", "Power / battery",
             "Sensor / optics", "Mechanical failure", "Software / firmware", "Other"]
MALFUNCTIONS = [(MAL_NONE, GREEN)] + [(m, RED) for m in MAL_TYPES]
MAL_IMPACT = [("No impact", GREEN), ("Minor (fixed quickly)", LIME, "Minor"),
              ("Degraded mission", YELLOW), ("Mission abort", ORANGE), ("Platform lost", RED)]


def opts(spec):
    """Normalise an option spec to [(value, color, label)]."""
    out = []
    for o in spec:
        if isinstance(o, tuple):
            v, c = o[0], o[1]
            out.append((v, c, o[2] if len(o) > 2 else str(v)))
        else:
            out.append((o, BLUE, str(o)))
    return out


def values(spec):
    return [v for v, *_ in opts(spec)]


def color_of(spec, value):
    for v, c, _ in opts(spec):
        if v == value:
            return c
    return GRAY


def label_of(spec, value):
    for v, _, lbl in opts(spec):
        if v == value:
            return lbl
    return "" if value is None else str(value)


# Which spec colors each stored field (review page, database view).
FIELD_SPECS = {
    "weather": WEATHER, "time_of_day": TIME_OF_DAY, "terrain": TERRAIN, "ew_level": THREAT,
    "ad_level": THREAT, "enemy_strength": ENEMY_STRENGTH, "enemy_adaptation": ENEMY_ADAPT,
    "outcome": OUTCOMES, "outcome_driver": OUTCOME_DRIVERS, "effectiveness": EFFECTIVENESS,
    "malfunction_impact": MAL_IMPACT, "platform_status": PLATFORM_STATUS,
    "field_modified": YES_NO_NEUTRAL, **{k: RATING for k, *_ in TRAITS},
}

# Required on level 1 (v3). Effectiveness is also required, once per platform card.
REQUIRED = {"report_date": "Report date", "unit_type": "Unit type", "task": "Primary task",
            "platforms": "At least one platform", "malfunctions": "Malfunction check",
            "outcome": "Mission outcome"}

MODEL_TIER = 1   # levels 2-4 are never shown or analysed here

# ---------------------------------------------------------------------------
# Database: schema and upgrade steps copied from field_report_app_v3.py
# ---------------------------------------------------------------------------
PLATFORM_COLUMNS = [
    ("platform_status", "TEXT"), ("effectiveness", "INTEGER"),
    ("rating_reliability", "INTEGER"), ("rating_ease", "INTEGER"),
    ("rating_sustainment", "INTEGER"), ("rating_survivability", "INTEGER"),
    ("rating_crew_survivability", "INTEGER"), ("rating_lethality", "INTEGER"),
    ("rating_hard_to_hit", "INTEGER"),
    ("field_modified", "INTEGER DEFAULT 0"),
    ("rated_individually", "INTEGER DEFAULT 0"),
]


def add_missing_columns(conn, table, columns):
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns:
        if name not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def reports_ddl(name="reports", if_not_exists=False):
    return f"""CREATE TABLE {'IF NOT EXISTS ' if if_not_exists else ''}{name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_date TEXT NOT NULL, unit_type TEXT, unit_id TEXT, task TEXT NOT NULL,
        platforms TEXT NOT NULL,
        malfunctions TEXT, malfunction_impact TEXT, field_fix TEXT,
        effectiveness INTEGER,
        rating_reliability INTEGER, rating_ease INTEGER,
        rating_sustainment INTEGER, rating_survivability INTEGER,
        supply TEXT, interoperability TEXT,
        platform_lost INTEGER DEFAULT 0, field_modified INTEGER DEFAULT 0,
        location TEXT, weather TEXT, time_of_day TEXT, terrain TEXT,
        enemy_type TEXT, enemy_strength TEXT, ew_level TEXT, ad_level TEXT, enemy_adaptation TEXT,
        outcome TEXT NOT NULL, notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        outcome_driver TEXT,
        tier INTEGER NOT NULL DEFAULT 1 CHECK (tier IN (1, 2, 3, 4)),
        tier_answers TEXT,
        CHECK (tier <> 1 OR (unit_type IS NOT NULL AND malfunctions IS NOT NULL AND effectiveness IS NOT NULL)))"""


def migrate_reports_to_tiers(conn):
    """Give an older reports table the tier columns (same rebuild as the desktop v3 app)."""
    old_cols = [r[1] for r in conn.execute("PRAGMA table_info(reports)")]
    if not old_cols or "tier" in old_cols:
        return False
    conn.commit()
    keep_isolation = conn.isolation_level
    conn.isolation_level = None
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN")
        conn.execute("DROP TABLE IF EXISTS reports_new")
        conn.execute(reports_ddl("reports_new"))
        new_cols = [r[1] for r in conn.execute("PRAGMA table_info(reports_new)")]
        common = ",".join(c for c in new_cols if c in old_cols)
        before = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        conn.execute(f"INSERT INTO reports_new ({common}) SELECT {common} FROM reports")
        if conn.execute("SELECT COUNT(*) FROM reports_new").fetchone()[0] != before:
            raise sqlite3.IntegrityError("report count changed during the tier upgrade")
        seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'reports'").fetchone()
        conn.execute("DROP TABLE reports")
        conn.execute("ALTER TABLE reports_new RENAME TO reports")
        if seq:
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'reports'")
            conn.execute("INSERT INTO sqlite_sequence (name, seq) VALUES ('reports', ?)", (seq[0],))
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.isolation_level = keep_isolation
    return True


def connect(path=None):
    path = Path(path or DB_PATH)
    made_v3_backup = False
    if path.exists():   # same one-time safety copies as the desktop app before upgrading an older file
        probe = sqlite3.connect(path)
        have = {r[1] for r in probe.execute("PRAGMA table_info(report_platforms)")}
        rcols = {r[1] for r in probe.execute("PRAGMA table_info(reports)")}
        probe.close()
        backup = path.with_name("field_reports_pre_v3.bak")
        if have and "rated_individually" not in have and not backup.exists():
            shutil.copy2(path, backup)
            made_v3_backup = True
        tier_backup = path.with_name("field_reports_pre_tiers.bak")
        if rcols and "tier" not in rcols and not made_v3_backup and not tier_backup.exists():
            shutil.copy2(path, tier_backup)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(reports)")]
    if cols and "platforms" not in cols:
        conn.execute("DROP TABLE IF EXISTS reports_v1_backup")
        conn.execute("ALTER TABLE reports RENAME TO reports_v1_backup")
    elif cols and "location" not in cols:
        conn.execute("ALTER TABLE reports ADD COLUMN location TEXT")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS weapons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL, category TEXT NOT NULL, origin TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS report_platforms (
            report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
            platform TEXT NOT NULL);
    """)
    conn.execute(reports_ddl("reports", if_not_exists=True))
    first_upgrade = "rated_individually" not in {
        r[1] for r in conn.execute("PRAGMA table_info(report_platforms)")}
    add_missing_columns(conn, "reports", [("outcome_driver", "TEXT")])
    add_missing_columns(conn, "report_platforms", PLATFORM_COLUMNS)
    if first_upgrade:
        copy = ", ".join(f"{c} = (SELECT r.{c} FROM reports r WHERE r.id = report_platforms.report_id)"
                         for c in ("effectiveness", "rating_reliability", "rating_ease",
                                   "rating_sustainment", "rating_survivability", "field_modified"))
        conn.execute(f"UPDATE report_platforms SET {copy}")
    migrate_reports_to_tiers(conn)
    if conn.execute("SELECT COUNT(*) FROM weapons").fetchone()[0] == 0:
        conn.executemany("INSERT INTO weapons (name, category, origin) VALUES (?,?,?)",
                         [(n, c, o) for c, items in CATALOG.items() for n, o in items])
    conn.commit()
    return conn


def rollup(plat_rows):
    """Report-level summary of the per-platform ratings (rounded mean), as in v3."""
    def mean(key):
        vals = [p[key] for p in plat_rows if p.get(key)]
        return int(sum(vals) / len(vals) + 0.5) if vals else None
    return {"effectiveness": mean("effectiveness"), "rating_reliability": mean("rating_reliability"),
            "rating_ease": mean("rating_ease"), "rating_sustainment": mean("rating_sustainment"),
            "rating_survivability": mean("rating_survivability"),
            "platform_lost": int(any(p.get("platform_status") == STATUS_LOST for p in plat_rows)),
            "field_modified": int(any(p.get("field_modified") for p in plat_rows))}


def insert_report(conn, rec, plat_rows):
    """rec = report-level fields; plat_rows = one dict per platform (name, status, ratings)."""
    rec = dict(rec, platforms="; ".join(p["platform"] for p in plat_rows))
    rec.setdefault("tier", MODEL_TIER)
    keys = list(rec.keys())
    cur = conn.execute(f"INSERT INTO reports ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
                       [rec[k] for k in keys])
    conn.executemany(
        "INSERT INTO report_platforms (report_id, platform, platform_status, effectiveness, "
        "rating_reliability, rating_ease, rating_sustainment, rating_survivability, "
        "rating_crew_survivability, rating_lethality, rating_hard_to_hit, field_modified, "
        "rated_individually) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)",
        [(cur.lastrowid, p["platform"], p.get("platform_status"), p.get("effectiveness"),
          p.get("rating_reliability"), p.get("rating_ease"), p.get("rating_sustainment"),
          p.get("rating_survivability"), p.get("rating_crew_survivability"), p.get("rating_lethality"),
          p.get("rating_hard_to_hit"), int(bool(p.get("field_modified")))) for p in plat_rows])
    conn.commit()
    return cur.lastrowid


def delete_reports(conn, ids):
    conn.executemany("DELETE FROM reports WHERE id = ?", [(int(i),) for i in ids])
    conn.commit()


def reports_df(conn):
    """Level 1 reports only."""
    return pd.read_sql_query(
        "SELECT * FROM reports WHERE tier = ? ORDER BY report_date DESC, id DESC", conn,
        params=(MODEL_TIER,))


def other_level_count(conn):
    return conn.execute("SELECT COUNT(*) FROM reports WHERE tier <> ?", (MODEL_TIER,)).fetchone()[0]


def platform_rows_df(conn, report_id=None):
    """One row per platform card (level 1), with category/origin and the report's fields.
    Card columns are prefixed p_ so they don't clash with the report-level rollup."""
    sql = ("SELECT rp.platform, COALESCE(w.category, 'Unlisted / custom') AS category, "
           "COALESCE(w.origin, 'Custom') AS origin, rp.platform_status AS p_status, "
           "rp.effectiveness AS p_effectiveness, "
           + ", ".join(f"rp.{k} AS p_{k}" for k, *_ in TRAITS)
           + ", rp.field_modified AS p_field_modified, rp.rated_individually, r.* "
           "FROM report_platforms rp JOIN reports r ON r.id = rp.report_id "
           "LEFT JOIN weapons w ON w.name = rp.platform WHERE r.tier = ?")
    params = [MODEL_TIER]
    if report_id is not None:
        sql += " AND r.id = ?"
        params.append(int(report_id))
    return pd.read_sql_query(sql + " ORDER BY rp.rowid", conn, params=params)


# SYNTHETIC sample data, ported from field_report_app_v3.py (level 1 only).
SAMPLE_SECTOR_WEIGHTS = {
    "Pokrovsk": 6, "Kupiansk": 4.5, "Chasiv Yar / Toretsk": 4, "Huliaipole": 3.5,
    "Kharkiv": 2, "Kramatorsk / Sloviansk": 2, "Lyman / Siversk": 1.5, "Kurakhove / Vuhledar": 1.5,
    "Orikhiv / Zaporizhzhia": 1.2, "Kherson / Dnipro River": 1, "Rear area / deep strike": 0.8,
    "Black Sea / Odesa": 0.5, "Northern border (Sumy)": 0.4}


def generate_sample_data(conn, n=150, seed=None):
    """SYNTHETIC. Enemy EW lowers results for EW-sensitive categories and causes more comms / EW
    malfunctions. Each platform gets its own result and ratings; about 1 report in 4 skips the
    optional adversary section."""
    rng = random.Random(seed)
    items = conn.execute("SELECT name, category FROM weapons").fetchall()
    ew_idx = {"Unknown": 0, "None": 0, "Low": 1, "Medium": 2, "High": 3}

    def rate(base):
        return max(1, min(5, round(base + rng.gauss(0, 0.8))))

    for _ in range(n):
        picks = rng.sample(items, rng.choice([1, 1, 2, 3]))
        sens_mean = sum(EW_SENSITIVITY.get(c, 0.1) for _, c in picks) / len(picks)
        ew = rng.choice(["None", "Low", "Medium", "High"])
        p_mal = 0.12 + 0.13 * ew_idx[ew] * sens_mean
        mals = [MAL_NONE]
        if rng.random() < p_mal:
            if sens_mean > 0.5 and ew_idx[ew] >= 2 and rng.random() < 0.7:
                mals = rng.sample(["EW / GPS interference", "Comms / data-link loss"], rng.choice([1, 2]))
            else:
                mals = rng.sample(MAL_TYPES[:-1], rng.choice([1, 1, 2]))
        has_mal = mals != [MAL_NONE]
        plat_rows, scores = [], []
        for name, cat in picks:
            sens = EW_SENSITIVITY.get(cat, 0.1)
            score = 4.4 - 0.6 * ew_idx[ew] * sens * 1.6 + rng.gauss(0, 0.7)
            scores.append(score)
            eff = max(1, min(5, round(score)))
            status = ("Performed as expected" if eff >= 4 else "Degraded but usable" if eff == 3
                      else STATUS_LOST if rng.random() < 0.4 else "Failed to perform")
            plat_rows.append({
                "platform": name, "platform_status": status, "effectiveness": eff,
                "rating_ease": max(1, rate(eff) - 1) if has_mal else rate(eff),
                "rating_sustainment": rate(eff), "rating_survivability": rate(eff),
                "rating_crew_survivability": rate(eff), "rating_lethality": rate(eff),
                "rating_hard_to_hit": rate(eff), "field_modified": int(rng.random() < 0.25)})
        mean_eff = max(1, min(5, round(sum(scores) / len(scores))))
        impact_vals = values(MAL_IMPACT)
        impact = (rng.choice(impact_vals[1:]) if has_mal and mean_eff <= 3
                  else "Minor (fixed quickly)" if has_mal else "No impact")
        outcome = ("Success" if mean_eff >= 4 else "Partial success" if mean_eff == 3
                   else rng.choice(["Failure", "Failure", "Aborted"]))
        driver = None
        if outcome != "Success":
            if sens_mean > 0.5 and ew_idx[ew] >= 2 and rng.random() < 0.6:
                driver = "Enemy EW / countermeasures"
            elif has_mal and rng.random() < 0.5:
                driver = "Equipment performance / malfunction"
            else:
                driver = rng.choice(values(OUTCOME_DRIVERS))
        enemy = {"enemy_type": rng.choice(ENEMY_TYPES),
                 "enemy_strength": rng.choice(values(ENEMY_STRENGTH)),
                 "ad_level": rng.choice(values(THREAT)),
                 "enemy_adaptation": rng.choice(values(ENEMY_ADAPT))}
        if rng.random() < 0.25:
            enemy = {k: None for k in ADVERSARY_KEYS}
        enemy["ew_level"] = ew
        rec = {
            "report_date": (date.today() - timedelta(days=rng.randint(0, 60))).isoformat(),
            "unit_type": rng.choice(UNIT_TYPES[:-1]), "unit_id": f"U-{rng.randint(100, 999)}",
            "task": rng.choice(TASKS[:-1]),
            "malfunctions": "; ".join(mals), "malfunction_impact": impact,
            "location": rng.choices(list(SAMPLE_SECTOR_WEIGHTS),
                                    weights=list(SAMPLE_SECTOR_WEIGHTS.values()))[0],
            "weather": rng.choice(values(WEATHER)), "time_of_day": rng.choice(values(TIME_OF_DAY)),
            "terrain": rng.choice(values(TERRAIN)),
            "outcome": outcome, "outcome_driver": driver, "notes": "[sample data]",
            "tier": MODEL_TIER, **enemy, **rollup(plat_rows),
        }
        insert_report(conn, rec, plat_rows)
