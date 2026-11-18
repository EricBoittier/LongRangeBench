"""anywidget wrapper around cw_scenes.js: the ChargeWorld notebook's three.js scenes.

    import marimo as mo, cw_widgets as W
    view = mo.ui.anywidget(W.Scene("explorer", data={...}))

three.js is the vendored, pinned r170 build in vendor/three-0.170.0/ (sha256 in SOURCE.txt). It is embedded in the
widget module and imported from a Blob URL, so the scenes work offline, in `marimo edit`, `marimo run` and in a static
`marimo export html` file. If the vendored file is missing, the module falls back to the same version on jsDelivr.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anywidget
import traitlets

HERE = Path(__file__).resolve().parent
THREE = HERE / "vendor" / "three-0.170.0" / "three.module.min.js"
THREE_SHA256 = "08fd7545d13d2c7fb65ab691530a802dafefd638596501854f267d0fb13c39e7"
CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js"

GRAMMAR = {
    "neg": "#2A6BC4", "zero": "#F5F5F5", "pos": "#BF382B", "ink": "#1A1A1A",
    "greys": ["#4D4D4D", "#8C8C8C", "#BDBDBD", "#E3E3E3"], "card": "#F7F7F5", "accent": "#E69F00", "trap": "#6C4F9E", "ok": "#2E8B57",
    "font": "Arimo, 'Liberation Sans', Arial, sans-serif",
}


def _loader() -> str:
    if THREE.exists():
        src = THREE.read_text()
        if hashlib.sha256(src.encode()).hexdigest() != THREE_SHA256:
            raise RuntimeError(f"{THREE} does not match the pinned sha256")
        return ("const THREE_SRC = " + json.dumps(src) + ";\n"
                "function loadThree() { return import(URL.createObjectURL(new Blob([THREE_SRC], "
                "{ type: 'text/javascript' }))); }\n")
    return f"function loadThree() {{ return import({json.dumps(CDN)}); }}\n"


_ESM = _loader() + (HERE / "cw_scenes.js").read_text()


class Scene(anywidget.AnyWidget):
    _esm = _ESM
    scene = traitlets.Unicode("explorer").tag(sync=True)
    data = traitlets.Any({}).tag(sync=True)
    grammar = traitlets.Dict(GRAMMAR).tag(sync=True)
    params = traitlets.Dict({}).tag(sync=True)

    def __init__(self, scene: str, data=None, **params):
        super().__init__(scene=scene, data=data or {}, grammar=GRAMMAR, params={"width": 760, **params})
