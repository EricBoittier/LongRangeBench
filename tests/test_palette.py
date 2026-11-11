"""The palette carries meaning, so the properties that make it readable are tested, not just eyeballed.

Two invariants matter, and both have already been broken once:

* **No two model families share a hue.** The family hue is the first thing read off a chart, and a repeat would
  silently merge two models' identities.
* **No two members of one family share a mark.** Within a family the hue is deliberately the same, so the marker is
  the *only* thing separating members. Markers are assigned by cycling a list, so a family with more members than
  the list has entries wraps and produces two models that are identical in both channels. The PET family has eight.
"""
import lrbench.models as M
from lrbench.palette import FAMILIES, GROUPS, MARKERS, family_of, style_of

#: Every name the project actually plots: the released models plus the oracle row.
NAMES = sorted(set(M.MODELS) | {"oracle"})


def test_no_two_families_share_a_hue():
    hues = {f: spec["colour"] for f, spec in FAMILIES.items()}
    assert len(set(hues.values())) == len(hues), f"two families share a hue: {hues}"


def test_task_groups_are_achromatic_so_they_cannot_collide_with_a_model():
    """Structure is achromatic, data is chromatic. The task-group labels are structure, so they carry no chroma --
    which is what *guarantees* they can never be mistaken for a model family, rather than merely happening not to."""
    family_hues = {spec["colour"].lower() for spec in FAMILIES.values()}
    for key, g in GROUPS.items():
        h = g["colour"].lstrip("#")
        r, gr, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        assert r == gr == b, f"task group {key} is not grey: {g['colour']}"
        assert g["colour"].lower() not in family_hues


def test_every_model_resolves_to_a_real_family():
    """A model that falls through to 'other' is drawn in the fallback grey, which reads as 'not a model'."""
    orphans = [n for n in NAMES if family_of(n) == "other"]
    assert not orphans, f"no family hue for {orphans}"


def test_members_of_a_family_are_distinguishable():
    seen = {}
    for n in NAMES:
        st = style_of(n, NAMES)
        key = (st["colour"], st["marker"])
        assert key not in seen, f"{n} and {seen[key]} share both hue and marker"
        seen[key] = n


def test_there_are_enough_markers_for_the_largest_family():
    sizes = {}
    for n in NAMES:
        sizes[family_of(n)] = sizes.get(family_of(n), 0) + 1
    biggest = max(sizes.values())
    assert len(MARKERS) >= biggest, (
        f"the largest family has {biggest} members but only {len(MARKERS)} markers exist, so the cycle wraps: "
        f"{sizes}")
