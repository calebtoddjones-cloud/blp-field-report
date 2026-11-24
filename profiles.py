"""Platform characteristics ("capability profile") for every catalog platform.

ESTIMATES, NOT AUTHORITATIVE. Coarse values from general public descriptions of each system (what it is,
how it moves, how it is guided or controlled), chosen so analysts can explore design trade-offs in the Lab.
Replace with sourced values (e.g. the U.S. Army ODIN Worldwide Equipment Guide) before using for anything
real. Nothing here comes from the reports; the form is unchanged.

Each profile: protection, signature, mobility, link, ew_hardening, reach, crew.
Ordinal scales are stored as integers (index into the level list); mobility and link are categories.
Platforms not listed here (custom ones a unit types in) get their class's most common profile.
"""

ATTRS = {
    # key: (label, levels or None for categorical, help)
    "protection": ("Armor / protection", ["None", "Light", "Medium", "Heavy"],
                   "None = soft-skin or no armor; Light = splinter / light armor; Medium = APC / IFV; Heavy = tank."),
    "signature": ("Signature (size, heat, emissions)", ["Small", "Medium", "Large"],
                  "How easy it is to spot: person-portable or small drone, vehicle-size, or large / radiating."),
    "ew_hardening": ("Jamming resistance", ["None", "Some", "Hardened"],
                     "Anti-jam GPS, frequency hopping, or a link that cannot be jammed."),
    "reach": ("Reach", ["Close <5 km", "Short 5–40 km", "Medium 40–150 km", "Long >150 km"],
              "Rough engagement or operating range."),
    "crew": ("Crew exposed", ["Uncrewed / remote", "1–2", "3–5", "6+"],
             "People on or at the platform when it fights (operators of remote systems count as uncrewed)."),
    "mobility": ("Mobility", None, "How it moves."),
    "link": ("Guidance / control link", None, "What it depends on to find, guide or communicate."),
}
ORDINAL = [k for k, (_, lv, _) in ATTRS.items() if lv]
CATEGORICAL = [k for k, (_, lv, _) in ATTRS.items() if not lv]
MOBILITY = ["Man-portable", "Towed / emplaced", "Wheeled", "Tracked", "Airborne", "Naval", "Munition"]
LINKS = ["None / unguided", "Radio data link", "Inertial / GPS", "Wire / fiber-optic", "Radar", "IR / EO seeker",
         "Laser", "Satcom"]
CHOICES = {"mobility": MOBILITY, "link": LINKS}
ORDER = ["protection", "signature", "mobility", "link", "ew_hardening", "reach", "crew"]

M, T, W, TR, A, N, MU = MOBILITY
NONE, RADIO, GPS, WIRE, RADAR, IR, LASER, SAT = LINKS

# name: (protection, signature, mobility, link, ew_hardening, reach, crew)
PROFILES = {
    # Loitering / Strike UAS
    "Switchblade 300": (0, 0, A, RADIO, 1, 1, 0), "Switchblade 600": (0, 0, A, RADIO, 1, 1, 0),
    "Phoenix Ghost": (0, 0, A, RADIO, 1, 1, 0), "Hero-120": (0, 0, A, RADIO, 1, 1, 0),
    "Warmate": (0, 0, A, RADIO, 1, 1, 0), "Liutyi long-range UAS": (0, 1, A, GPS, 1, 3, 0),
    "RAM II strike UAS": (0, 0, A, RADIO, 0, 1, 0),
    # FPV / Small Strike UAS
    "FPV 7-inch": (0, 0, A, RADIO, 0, 0, 0), "FPV 10-inch heavy": (0, 0, A, RADIO, 0, 0, 0),
    "Fiber-optic FPV": (0, 0, A, WIRE, 2, 0, 0), "Drop-munition quadcopter": (0, 0, A, RADIO, 0, 0, 0),
    "Commercial quad (modified)": (0, 0, A, RADIO, 0, 0, 0),
    # Recon / ISR UAS
    "RQ-20 Puma": (0, 0, A, RADIO, 1, 1, 0), "RQ-11 Raven": (0, 0, A, RADIO, 0, 1, 0),
    "ScanEagle": (0, 0, A, RADIO, 1, 2, 0), "Leleka-100": (0, 0, A, RADIO, 1, 1, 0),
    "Furia": (0, 0, A, RADIO, 1, 1, 0), "Shark": (0, 0, A, RADIO, 1, 2, 0),
    "Bayraktar TB2": (0, 1, A, RADIO, 1, 2, 0), "Commercial recon quad": (0, 0, A, RADIO, 0, 0, 0),
    # Counter-UAS
    "Coyote Block 2": (0, 0, A, RADAR, 1, 1, 0), "Gepard SPAAG": (1, 2, TR, RADAR, 1, 0, 2),
    "Interceptor drone": (0, 0, A, RADIO, 0, 0, 0), "Acoustic sensor network": (0, 0, T, RADIO, 1, 1, 0),
    "Mobile fire group (MG)": (0, 1, W, NONE, 2, 0, 2),
    # Air Defense
    "Patriot PAC-3": (0, 2, W, RADAR, 2, 2, 3), "NASAMS": (0, 1, W, RADAR, 2, 1, 2),
    "IRIS-T SLM": (0, 1, W, RADAR, 2, 1, 2), "Stinger (MANPADS)": (0, 0, M, IR, 1, 0, 1),
    "Avenger": (1, 1, W, IR, 1, 0, 1), "HAWK": (0, 2, T, RADAR, 1, 1, 3),
    "Buk-M1": (1, 2, TR, RADAR, 1, 1, 2), "S-300": (0, 2, W, RADAR, 1, 2, 3),
    # Anti-Armor
    "Javelin": (0, 0, M, IR, 2, 0, 1), "TOW": (0, 1, M, WIRE, 2, 0, 2), "AT4": (0, 0, M, NONE, 2, 0, 1),
    "NLAW": (0, 0, M, GPS, 2, 0, 1), "Stugna-P": (0, 1, M, LASER, 2, 0, 2), "Skif": (0, 1, M, LASER, 2, 0, 2),
    # Artillery / Mortars
    "M777 155mm": (0, 1, T, NONE, 2, 1, 3), "M109 Paladin": (1, 2, TR, NONE, 2, 1, 3),
    "M252 81mm mortar": (0, 0, M, NONE, 2, 0, 2), "PzH 2000": (1, 2, TR, NONE, 2, 1, 2),
    "CAESAR": (1, 1, W, NONE, 2, 1, 2), "Krab": (1, 2, TR, NONE, 2, 1, 2),
    "Bohdana 2S22": (1, 1, W, NONE, 2, 1, 2), "2S1 Gvozdika": (1, 1, TR, NONE, 2, 1, 2),
    "120mm mortar": (0, 0, T, NONE, 2, 0, 2),
    # Rockets / Missiles
    "HIMARS (GMLRS)": (1, 1, W, GPS, 1, 2, 1), "M270 MLRS": (1, 2, TR, GPS, 1, 2, 2),
    "ATACMS": (0, 0, MU, GPS, 1, 3, 0), "Storm Shadow": (0, 0, MU, IR, 2, 3, 0),
    "Neptune": (0, 1, MU, RADAR, 1, 3, 0), "FP-5 Flamingo": (0, 1, MU, GPS, 1, 3, 0),
    "BM-21 Grad": (0, 1, W, NONE, 2, 1, 2),
    # Precision Munitions
    "Excalibur 155mm": (0, 0, MU, GPS, 0, 1, 0), "JDAM-ER": (0, 0, MU, GPS, 0, 2, 0),
    "GLSDB": (0, 0, MU, GPS, 0, 2, 0),
    # Armored / Vehicles
    "M1A1 Abrams": (3, 2, TR, NONE, 2, 0, 2), "M2 Bradley": (2, 2, TR, NONE, 2, 0, 3),
    "Stryker": (2, 1, W, NONE, 2, 0, 3), "M113": (1, 1, TR, NONE, 2, 0, 3),
    "MaxxPro MRAP": (2, 1, W, NONE, 2, 0, 3), "HMMWV": (1, 1, W, NONE, 2, 0, 2),
    "Leopard 2A6": (3, 2, TR, NONE, 2, 0, 2), "Challenger 2": (3, 2, TR, NONE, 2, 0, 2),
    "T-64BV": (2, 2, TR, NONE, 2, 0, 2), "BMP-2": (1, 1, TR, NONE, 2, 0, 3),
    # Small Arms / Crew-Served
    "M4 carbine": (0, 0, M, NONE, 2, 0, 1), "M249 SAW": (0, 0, M, NONE, 2, 0, 1),
    "M240B": (0, 0, M, NONE, 2, 0, 1), "M2 .50 cal": (0, 0, M, NONE, 2, 0, 1),
    "M107 .50 sniper": (0, 0, M, NONE, 2, 0, 1), "AK-74": (0, 0, M, NONE, 2, 0, 1),
    # EW / C2 / Comms
    "Starlink terminal": (0, 0, T, SAT, 1, 3, 0), "Delta situational awareness": (0, 0, T, RADIO, 1, 3, 0),
    "Kvertus EW system": (0, 1, T, RADIO, 2, 0, 1), "Bukovel-AD": (0, 1, W, RADIO, 2, 1, 1),
    "Silent Archer EW": (0, 1, W, RADAR, 2, 1, 1), "AN/PRC-152 radio": (0, 0, M, RADIO, 1, 0, 1),
    "AN/PRC-117G radio": (0, 0, M, RADIO, 1, 1, 1),
    # Naval / USV
    "Magura V5": (0, 0, N, SAT, 1, 3, 0), "Sea Baby": (0, 0, N, SAT, 1, 3, 0),
    "Harpoon (coastal)": (0, 1, W, RADAR, 1, 2, 1),
}


def as_dict(t):
    return dict(zip(ORDER, t))


def class_default(category, catalog):
    """Most common value of each characteristic among the class's listed platforms."""
    rows = [as_dict(PROFILES[n]) for n, _ in catalog.get(category, []) if n in PROFILES]
    if not rows:
        return as_dict((0, 1, MOBILITY[2], LINKS[1], 1, 1, 1))
    return {k: max(set(r[k] for r in rows), key=[r[k] for r in rows].count) for k in ORDER}


def profile(name, category, catalog):
    """(profile dict, is_estimate_for_this_platform). Unlisted platforms get the class default."""
    if name in PROFILES:
        return as_dict(PROFILES[name]), True
    return class_default(category, catalog), False


def label(key, value):
    lv = ATTRS[key][1]
    return lv[int(value)] if lv else str(value)
