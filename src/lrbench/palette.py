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
    # tableau:tab10_new — Tableau 10. The house palette of this project (it is the one used in mmml), and the
    # default for model families. It is *not* colour-vision safe across all pairs: its red (#E15759) and green
    # (#59A14F) are indistinguishable to a deuteranope. That is tolerable here only because colour is never the
    # sole encoding of a model — the family name is printed beside every swatch and members differ by marker —
    # and it is why "okabe_ito" is kept below as a one-line switch for anyone who needs it.
    "tableau10": ["#4E79A7", "#F28E2B", "#76B7B2", "#E15759", "#B07AA1", "#9C755F", "#59A14F", "#EDC948",
                  "#FF9DA7", "#BAB0AB"],
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

#: The ramps the chemiscope views are *actually* drawn with, applied to Plotly's marker colorscale directly rather
#: than chosen from chemiscope's own fixed list. CMasher's maps are perceptually uniform and colour-vision safe,
#: and these two are taken from it -- but neither is used as published, for reasons that are about this page rather
#: than about the maps:
#:
#: ``magnitude`` — ``cmasher:rainforest`` sampled over **0.04-0.88** instead of 0-1. The full map runs from pure
#: black to pure white, and a ramp that ends at the page colour hides its own extreme: the largest values would be
#: white markers on a white background. Truncated, lightness still rises monotonically (0.01 -> 0.83), so it reads
#: correctly in greyscale.
#:
#: ``signed`` — ``cmasher:fusion`` sampled over **0.10-0.90**, with the neutral nudged 10 % toward ink. Two separate
#: problems: at the extremes fusion converges on nearly the same dark purple, so maximum positive and maximum
#: negative looked alike, which is fatal for a signed quantity; truncating leaves dark red against indigo at
#: matched lightness (L 0.13 vs 0.14). And its middle is pure white, so a *zero* error -- the thing a diverging map
#: exists to make visible -- would have disappeared into the page. #e6e7e7 reads as neutral instead of as absent,
#: which is what crameri:vik does with #ECE5E0 and why that map has been the project's diverging default.
#:
#: CMasher's diverging maps are otherwise mostly dark-middled (``redshift``, ``iceburn``: black at zero), which
#: suits a black background and not this one. That is a question of destination, not quality.
CMASHER = {
    "magnitude": ["#080206", "#2A0535", "#3E057A", "#2C3893", "#055C86", "#06767B", "#1A916B", "#49A74A",
                  "#96B605", "#D4BE46", "#F7D09C"],
    "signed": ["#720633", "#A32E28", "#C36021", "#D09B52", "#DCCFA8", "#e6e7e7", "#AFD9E1", "#60B5DC",
               "#4785DD", "#574CD0", "#550B98"],
    # Qualitative stays Tableau, which is what the project already uses for model families and what chemiscope's
    # own "tab10" is (the classic Tableau 10). A ramp has no business carrying an identity.
    "identity": QUALITATIVE["tableau10"],
}

#: What chemiscope can be asked for, by the same three jobs. Its palettes are fixed by its own build, so these are
#: the best available members of each kind rather than the maps above — named here so a view never picks a
#: sequential ramp for a signed quantity just because it looked nice.
CHEMISCOPE = {
    "identity": ["tab10", "tab20", "tab20b", "tab20c"],
    "magnitude": ["cividis", "viridis", "magma", "plasma", "inferno"],
    "signed": ["bwr", "seismic", "rwg", "brg"],
}

#: The stops chemiscope actually renders for the palette this project assigns to each job, so a control can show
#: the ramp it is about to apply instead of naming it.
CHEMISCOPE_STOPS = {
    "identity": ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"],
    "magnitude": ["#00224E", "#434E6C", "#7D7C78", "#BCAE6C", "#FEE838"],
    "signed": ["#0000FF", "#8080FF", "#FFFEFE", "#FF7E7E", "#FF0000"],
}

INK, GREY, FAINT = "#15171a", "#6e7073", "#e3e2df"

#: **Structure is achromatic; data is chromatic.** The benchmark's task groups are structure — they label the
#: page, they are not a measurement — so they are drawn in neutral greys (crameri:grayC). Model families are data,
#: so they take the chromatic palette above. Nothing in the two sets can be confused, because one has no chroma at
#: all; that is a guarantee, not a judgement about particular hues.
GROUPS = {
    "A": {"colour": "#2B2B2B", "title": "Asymptotics", "sub": "the far field"},
    "B": {"colour": "#4A4A4A", "title": "Response", "sub": "the reply to a distant charge"},
    "C": {"colour": "#6B6B6B", "title": "Electrostatic variables", "sub": "charges and moments"},
    "D": {"colour": "#8D8D8D", "title": "Many-body", "sub": "what pairs cannot reach"},
}

#: Model families, in the order they are assigned colours. A family shares a hue; members are told apart by marker
#: and line style, so the family reads at a glance and the member reads on inspection. Matching is by substring on
#: the result's name, longest first, so ``so3lr-tauri`` wins over ``so3lr``.
FAMILIES = {
    "oracle":  {"match": ("oracle",),            "colour": "#15171a", "label": "oracle / ceiling"},
    "mace":    {"match": ("mace-polar", "mace-h", "mace-omol", "maceles", "mace"),
                "colour": "#4E79A7", "label": "MACE"},
    "so3lr":   {"match": ("so3lr",),             "colour": "#F28E2B", "label": "SO3LR"},
    "orbmol":  {"match": ("orbmol", "orb"),      "colour": "#76B7B2", "label": "OrbMol"},
    "dpa4":    {"match": ("dpa4", "deepmd"),     "colour": "#E15759", "label": "DeePMD / DPA4"},
    "aimnet":  {"match": ("aimnet",),            "colour": "#B07AA1", "label": "AIMNet2"},
    "grace":   {"match": ("grace", "ace"),       "colour": "#9C755F", "label": "GRACE (ACE)"},
    "cace":    {"match": ("cace",),              "colour": "#59A14F", "label": "CACE"},
    "fairchem": {"match": ("uma", "esen"),       "colour": "#EDC948", "label": "fairchem"},
    # One family for the whole PET architecture: the charge-conditioned OMol checkpoints, the SPICE and mixed ones,
    # and the PET-MAD materials model all share the hue and separate by marker. #FF9DA7 is the next unused member of
    # tableau10, so it collides with nothing above.
    "pet":     {"match": ("pet",),               "colour": "#FF9DA7", "label": "PET"},
}

#: Marker and dash, cycled within a family so members of one family are separable without a second hue. There must
#: be at least as many markers as the largest family has members, or two models would come out with the same hue
#: *and* the same mark and be genuinely indistinguishable -- the PET family alone has eight. ``test_palette`` holds
#: that line, so extend this list rather than letting the cycle wrap.
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">", "p"]
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
