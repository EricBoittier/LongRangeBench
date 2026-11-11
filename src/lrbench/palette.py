"""Colour chosen by what the data *is*, not by taste.

Three jobs, three kinds of map, and they are not interchangeable:

====================  ==========================================  ==========================================
the data is…          the map is…                                 because
====================  ==========================================  ==========================================
an **identity**       qualitative — unordered, equally salient     a model family or an interaction class has
(which model, which   hues                                         no order, so a ramp would invent one
 class)
a **magnitude**       sequential — perceptually uniform,           equal steps in the number must look like
(a distance, an       monotone in lightness                        equal steps in the colour
 absolute error)
a **signed** value    diverging — two hues about a neutral         zero is a real landmark and must be
(an error with a      midpoint, used symmetrically                 visible as one; the two directions must
 direction)                                                        be told apart
====================  ==========================================  ==========================================

The values come from the ``cmap`` catalogue and keep its names, so any of them can be regenerated::

    from cmap import Colormap
    [Colormap("crameri:batlow")(i / 6).hex for i in range(7)]

They are baked in as hex rather than imported, because this module is read by a notebook that also runs as
WebAssembly, where adding a dependency to save a dozen lines is a bad trade.

The sequential and diverging choices are Fabio Crameri's ``batlow`` and ``vik``: perceptually uniform, ordered
correctly when printed in greyscale, and readable with any common colour-vision deficiency. The qualitative choice
is Okabe–Ito, which was designed for exactly that constraint.
"""
from __future__ import annotations

# ---------------------------------------------------------------- identity: qualitative, no implied order
QUALITATIVE = {
    # matplotlib:okabe_ito — eight hues designed to stay distinguishable under deuteranopia and protanopia
    "okabe_ito": ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
    # tol:muted — Paul Tol's muted set, for when more than eight are needed
    "tol_muted": ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255",
                  "#AA4499"],
    # colorbrewer:Dark2 — stronger, for small marks on a light surface
    "dark2": ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"],
}

# ---------------------------------------------------------------- magnitude: sequential, perceptually uniform
SEQUENTIAL = {
    "batlow": ["#011959", "#144D62", "#3C6D56", "#828231", "#D29343", "#FDAC9E", "#FACCFA"],   # crameri:batlow
    "deep": ["#FDFECC", "#A5DFA7", "#5DBAA4", "#488E9E", "#3E6495", "#3F396C", "#281A2C"],      # cmocean:deep
}

# ---------------------------------------------------------------- signed: diverging, symmetric about the middle
DIVERGING = {
    "vik": ["#001261", "#034481", "#307DA6", "#94BED2", "#ECE5E0", "#DBAA8D", "#C27041", "#912D06",
            "#590008"],                                                                          # crameri:vik
    "roma": ["#7E1700", "#A5681F", "#C8B455", "#C0EAC3", "#5DC1D3", "#287AB8", "#033198"],       # crameri:roma
}

#: What chemiscope can be asked for, by the same three jobs. Its palettes are fixed by its own build, so these are
#: the best available members of each kind rather than the maps above — named here so a view never picks a
#: sequential ramp for a signed quantity just because it looked nice.
CHEMISCOPE = {
    "identity": ["tab10", "tab20", "tab20b", "tab20c"],
    "magnitude": ["cividis", "viridis", "magma", "plasma", "inferno"],
    "signed": ["bwr", "seismic", "rwg", "brg"],
}

INK, GREY, FAINT = "#15171a", "#6e7073", "#e3e2df"

#: Model families, in the order they are assigned colours. A family shares a hue; members are told apart by marker
#: and line style, so the family reads at a glance and the member reads on inspection. Matching is by substring on
#: the result's name, longest first, so ``so3lr-tauri`` wins over ``so3lr``.
FAMILIES = {
    "oracle":  {"match": ("oracle",),            "colour": "#000000", "label": "oracle / ceiling"},
    "mace":    {"match": ("mace-polar", "mace-h", "mace-omol", "maceles", "mace"),
                "colour": "#0072B2", "label": "MACE"},
    "so3lr":   {"match": ("so3lr",),             "colour": "#009E73", "label": "SO3LR"},
    "orbmol":  {"match": ("orbmol", "orb"),      "colour": "#D55E00", "label": "OrbMol"},
    "dpa4":    {"match": ("dpa4", "deepmd"),     "colour": "#CC79A7", "label": "DeePMD / DPA4"},
    "aimnet":  {"match": ("aimnet",),            "colour": "#56B4E9", "label": "AIMNet2"},
    "grace":   {"match": ("grace", "ace"),       "colour": "#E69F00", "label": "GRACE (ACE)"},
    "cace":    {"match": ("cace",),              "colour": "#7570B3", "label": "CACE"},
    "fairchem": {"match": ("uma", "esen"),       "colour": "#1B9E77", "label": "fairchem"},
}

#: Marker and dash, cycled within a family so members of one family are separable without a second hue.
MARKERS = ["o", "s", "D", "^", "v", "P"]
DASHES = ["", "4 2", "1 2", "6 2 1 2"]


def family_of(model: str) -> str:
    """Which family a result name belongs to, or ``"other"``. Longest match wins, so a specific name beats a
    prefix it happens to contain."""
    name = (model or "").lower()
    best, best_len = "other", 0
    for fam, spec in FAMILIES.items():
        for m in spec["match"]:
            if m in name and len(m) > best_len:
                best, best_len = fam, len(m)
    return best


def style_of(model: str, members: list | None = None) -> dict:
    """Colour, marker and dash for one model.

    ``members`` is the full list of result names, so that variants inside a family get different markers in a
    stable order rather than by luck of iteration.
    """
    fam = family_of(model)
    colour = FAMILIES.get(fam, {}).get("colour", GREY)
    idx = 0
    if members:
        same = sorted(m for m in members if family_of(m) == fam)
        if model in same:
            idx = same.index(model)
    return {"family": fam, "label": FAMILIES.get(fam, {}).get("label", "other"), "colour": colour,
            "marker": MARKERS[idx % len(MARKERS)], "dash": DASHES[idx % len(DASHES)], "variant": idx}


def ramp(kind: str, name: str | None = None) -> list:
    """The hex stops of a map, by job: ``identity``, ``magnitude`` or ``signed``."""
    table = {"identity": QUALITATIVE, "magnitude": SEQUENTIAL, "signed": DIVERGING}[kind]
    return list(table[name] if name else next(iter(table.values())))


def sample(stops: list, t: float) -> str:
    """Linear interpolation along a list of hex stops; ``t`` in [0, 1]."""
    if not stops:
        return GREY
    t = 0.0 if t != t else max(0.0, min(1.0, float(t)))
    x = t * (len(stops) - 1)
    i = int(x)
    if i >= len(stops) - 1:
        return stops[-1]
    f = x - i
    a, b = stops[i].lstrip("#"), stops[i + 1].lstrip("#")
    out = [round(int(a[j:j + 2], 16) * (1 - f) + int(b[j:j + 2], 16) * f) for j in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(out)
