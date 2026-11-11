"""GSCDB137 external validation (Liang & Head-Gordon, arXiv:2508.13468v2; JCTC 21, 12601 (2025)).

Repository JiashuLiang/GSCDB, pinned commit 0624edaad9a694735a13cd5493e2bb63a27e0fd7 (2026-09-05; its web title says
GSCDB138, the data and README say GSCDB137 v1.0.0). Everything read from the clone is untrusted data: run with
`python -I`, nothing in the clone is imported or executed.

    python -I -m lrbench.gscdb manifest REPO OUTDIR [--subsets ...]      -> manifest.json, MANIFEST.md, calcs.jsonl
    python -I -m lrbench.gscdb export   OUTDIR/manifest.json SELECTION OUT.xyz   (extxyz with charge/spin for lrbench)
    python -I -m lrbench.gscdb points   OUTDIR/manifest.json SELECTION OUT.jsonl (ORCA mkcalc.py input lines)
    python -I -m lrbench.gscdb score    OUTDIR/manifest.json PRED_DIR ORCA.jsonl OUT.json

Conventions (fixed here, recorded in the manifest):
- Reference energies are the repository's `Info/DatasetEval.csv` values in Hartree; every reaction is a linear
  combination (stoichiometry) of molecule total energies. The same combination is applied to model and ωB97M-V
  energies, so the comparison is always reaction-for-reaction.
- Two targets that are never interchanged:
    surrogate fidelity  dE_model - dE_wB97M-V/def2-TZVPD (OMol25 ORCA protocol, on the published geometry);
    end-to-end          dE_model - dE_GSCDB (model error + functional/basis error; never called "ML error").
- Fragments: taken from the reaction's own monomer terms (formal fragment charge/multiplicity as published). The atom
  map is coordinate matching (monomers frozen at the complex geometry) or, if the monomer is displaced, an element-
  sequence block match with a rigid-body (Kabsch) RMSD; both are recorded. Our own convention is computed alongside:
  covalent components (1.2 x covalent radii, metals/rare gases non-bonding; prep_omol25_val.comps rule) and the closest
  atom-atom distance between source fragments.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, OrderedDict

import numpy as np

HA_EV = 27.211386245988
HA_KCAL = 627.509474
PINNED_COMMIT = "0624edaad9a694735a13cd5493e2bb63a27e0fd7"

SYM = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
       "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
       "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
       "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
       "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]
ZOF = {s: i for i, s in enumerate(SYM)}
# Alvarez 2008 covalent radii (ase.data.covalent_radii), H..Rn
_COV = [0.2, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02,
        1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
        2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44,
        2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62,
        1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50]
NOBOND = {2, 3, 10, 11, 12, 18, 19, 20, 36, 37, 38, 54, 55, 56, 86}

P0 = ["IHB100x2", "AHB21", "CHB6", "IL16", "S66Rel7", "3BHET"]
P1 = ["NBC10", "BzDC215", "O24x4", "X40x5", "RG10N", "CT20"]
CATEGORY = {"IL16": "ion pair", "AHB21": "anion-neutral", "CHB6": "cation-neutral", "S66Rel7": "neutral PEC",
            "3BHET": "three-body", "NBC10": "neutral PEC", "BzDC215": "neutral PEC", "RG10N": "rare-gas PEC"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------- reading
def parse_mol(rec: dict):
    rows = [l.split() for l in rec["molecule"].strip().split("\n") if l.strip()]
    Z = np.array([ZOF[r[0].capitalize()] for r in rows], int)
    R = np.array([[float(x) for x in r[1:4]] for r in rows])
    return Z, R


def read_qchem_in(path: str):
    """$molecule block of a Q-Chem input: charge, multiplicity, Z, R (the README's validated primary format)."""
    t = open(path, errors="replace").read()
    m = re.search(r"\$molecule\s*\n(.*?)\$end", t, re.S | re.I)
    if not m:
        return None
    L = [l.split() for l in m.group(1).strip().split("\n") if l.strip()]
    q, mult = int(L[0][0]), int(L[0][1])
    Z = np.array([ZOF[r[0].capitalize()] for r in L[1:]], int)
    R = np.array([[float(x) for x in r[1:4]] for r in L[1:]])
    return q, mult, Z, R


def read_repo(repo: str):
    mols = json.load(open(os.path.join(repo, "Allmols_info.json")))
    rx = OrderedDict()
    with open(os.path.join(repo, "Info", "DatasetEval.csv")) as f:
        for row in csv.DictReader(f):
            s = row["Stoichiometry"].split(",")
            terms = [(float(s[i]), s[i + 1]) for i in range(0, len(s), 2)]
            rx[row["Reaction"]] = dict(set=row["Dataset"], ref_ha=float(row["Reference"]), terms=terms)
    sets = {}
    with open(os.path.join(repo, "Info", "Datasets.csv")) as f:
        for row in csv.DictReader(f):
            sets[row["Name"]] = row
    stderr = {}
    p = os.path.join(repo, "Info", "Standard_errors.csv")
    if os.path.exists(p):
        with open(p) as f:
            rd = csv.reader(f); head = next(rd)
            for row in rd:
                stderr[row[0]] = row[1:]
        stderr["_header"] = head
    E = {}
    with open(os.path.join(repo, "Analysis", "Molecule_Energies.csv")) as f:
        rd = csv.DictReader(f)
        for row in rd:
            E[row[""] if "" in row else row["Unnamed: 0"]] = row
    mae = {}
    with open(os.path.join(repo, "Analysis", "Errors_per_set_MAE.csv")) as f:
        for row in csv.DictReader(f):
            mae[row["Dataset"]] = row
    return mols, rx, sets, stderr, E, mae


# --------------------------------------------------------------------------------------------------------- geometry
def comps(Z, R):
    """Covalent components (our cluster split): 1.2 x (r_i + r_j), NOBOND elements never bonded."""
    n = len(Z)
    D = np.linalg.norm(R[:, None] - R[None], axis=-1)
    rc = np.array([_COV[z] for z in Z])
    B = D < 1.2 * (rc[:, None] + rc[None])
    nb = np.isin(Z, list(NOBOND)); B[nb, :] = False; B[:, nb] = False
    np.fill_diagonal(B, False)
    lab = -np.ones(n, int); c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        st = [s]; lab[s] = c
        while st:
            i = st.pop()
            for j in np.nonzero(B[i] & (lab < 0))[0]:
                lab[j] = c; st.append(j)
        c += 1
    return lab, c


def min_frag_distance(R, lab):
    k = sorted(set(lab.tolist()))
    if len(k) < 2:
        return None
    D = np.linalg.norm(R[:, None] - R[None], axis=-1)
    D[lab[:, None] == lab[None]] = np.inf
    return float(D.min())


def kabsch_rmsd(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    S[-1] *= d
    return float(np.sqrt(max(0.0, (np.sum(A ** 2) + np.sum(B ** 2) - 2 * S.sum()) / len(A))))


def match_fragment(Zc, Rc, Zm, Rm, used):
    """Atom indices of monomer (Zm, Rm) inside complex (Zc, Rc). 1) exact coordinates (frozen monomer) 2) contiguous
    element-sequence block (displaced rigid or relaxed monomer) with Kabsch RMSD. Returns (idx, method, dev)."""
    D = np.linalg.norm(Rm[:, None] - Rc[None], axis=-1)
    j = D.argmin(1)
    if D[np.arange(len(Zm)), j].max() < 1e-4 and len(set(j.tolist())) == len(j) and np.all(Zc[j] == Zm) \
            and not (set(j.tolist()) & used):
        return j.tolist(), "coordinates", float(D[np.arange(len(Zm)), j].max())
    n = len(Zm); best = None
    for s in range(len(Zc) - n + 1):
        idx = list(range(s, s + n))
        if set(idx) & used or not np.all(Zc[idx] == Zm):
            continue
        r = kabsch_rmsd(Rc[idx], Rm)
        if best is None or r < best[2]:
            best = (idx, "element-block", r)
    return best if best else (None, "unmatched", None)


# ---------------------------------------------------------------------------------------------------- basis counts
def tzvpd_counts(bas_path: str) -> dict:
    """Spherical basis-function count per element, from a def2-TZVPD basis as printed by ORCA."""
    L = {"S": 1, "P": 3, "D": 5, "F": 7, "G": 9, "H": 11}
    out, el = {}, None
    longnames = {"HYDROGEN": "H", "HELIUM": "He", "LITHIUM": "Li", "BERYLLIUM": "Be", "BORON": "B", "CARBON": "C",
                 "NITROGEN": "N", "OXYGEN": "O", "FLUORINE": "F", "NEON": "Ne", "SODIUM": "Na", "MAGNESIUM": "Mg",
                 "ALUMINUM": "Al", "ALUMINIUM": "Al", "SILICON": "Si", "PHOSPHORUS": "P", "SULFUR": "S",
                 "CHLORINE": "Cl", "ARGON": "Ar", "POTASSIUM": "K", "CALCIUM": "Ca", "BROMINE": "Br",
                 "KRYPTON": "Kr", "IODINE": "I", "XENON": "Xe", "SELENIUM": "Se", "ARSENIC": "As",
                 "GERMANIUM": "Ge", "GALLIUM": "Ga", "ZINC": "Zn", "COPPER": "Cu", "NICKEL": "Ni", "COBALT": "Co",
                 "IRON": "Fe", "MANGANESE": "Mn", "CHROMIUM": "Cr", "VANADIUM": "V", "TITANIUM": "Ti",
                 "SCANDIUM": "Sc", "RUBIDIUM": "Rb", "STRONTIUM": "Sr", "CESIUM": "Cs", "BARIUM": "Ba",
                 "TELLURIUM": "Te", "ANTIMONY": "Sb", "TIN": "Sn", "INDIUM": "In", "SILVER": "Ag", "PALLADIUM": "Pd",
                 "GOLD": "Au", "PLATINUM": "Pt", "MERCURY": "Hg", "LEAD": "Pb", "BISMUTH": "Bi", "CADMIUM": "Cd"}
    inside = False
    for line in open(bas_path, errors="replace"):
        t = line.strip()
        if t.startswith("$DATA"):
            inside = True; continue
        if not inside or not t:
            continue
        if t.startswith("$END"):
            break
        w = t.split()
        if w[0].upper() in longnames and len(w) == 1:
            el = longnames[w[0].upper()]; out[el] = 0; continue
        if el and w[0].upper() in L and len(w) == 2 and w[1].isdigit():
            out[el] += L[w[0].upper()]
    return out


# ------------------------------------------------------------------------------------------------------ manifest
def classify_set(name):
    return CATEGORY.get(name, "other")


def build_manifest(repo: str, subsets, bas_path: str, cross_check_qchem: bool = True):
    mols, rx, sets, stderr, Emol, mae = read_repo(repo)
    nbf_el = tzvpd_counts(bas_path) if bas_path and os.path.exists(bas_path) else {}
    files = {p: sha256(os.path.join(repo, p)) for p in ["Allmols_info.json", "Info/DatasetEval.csv",
                                                      "Info/Datasets.csv", "Info/Standard_errors.csv",
                                                      "Analysis/Molecule_Energies.csv"]}
    man = dict(source=dict(name="GSCDB137", arxiv="2508.13468v2", doi="10.1021/acs.jctc.5c01380",
                           repo="https://github.com/JiashuLiang/GSCDB", commit=PINNED_COMMIT,
                           commit_date="2026-09-05", note="repository web title says GSCDB138; README/data say "
                           "GSCDB137 v1.0.0", sha256=files),
               units=dict(reference="Hartree (DatasetEval.csv); converted to meV in outputs"),
               conventions=__doc__.split("Conventions (fixed here, recorded in the manifest):")[1].strip(),
               subsets={}, molecules={}, problems=[])
    wcol = "wB97M-V"
    for S in subsets:
        R_ = {k: v for k, v in rx.items() if v["set"] == S}
        info = sets.get(S, {})
        entries, keys = [], []
        for rid, r in R_.items():
            for c, k in r["terms"]:
                if k not in keys:
                    keys.append(k)
            # balance checks
            comp, chg = Counter(), 0.0
            for c, k in r["terms"]:
                Z, _ = parse_mol(mols[k])
                for z in Z:
                    comp[int(z)] += c
                chg += c * mols[k]["charge"]
            balanced = all(abs(v) < 1e-9 for v in comp.values())
            e = dict(id=rid, set=S, ref_ha=r["ref_ha"], ref_meV=r["ref_ha"] * HA_EV * 1000,
                     ref_kcal=r["ref_ha"] * HA_KCAL, terms=r["terms"], composition_balanced=balanced,
                     charge_balanced=abs(chg) < 1e-9)
            # repo's own wB97M-V/def2-QZVPPD reaction energy (sanity reference; not target (1))
            try:
                e["wb97mv_qzvppd_ha"] = sum(c * float(Emol[k][wcol]) for c, k in r["terms"])
            except (KeyError, ValueError):
                e["wb97mv_qzvppd_ha"] = None
            entries.append(e)
        # per-entry structure roles + fragment maps
        for e in entries:
            pos = [k for c, k in e["terms"] if c > 0]
            neg = [k for c, k in e["terms"] if c < 0]
            if S == "3BHET":
                cplx = [k for k in pos if k.endswith("_trim")][0]
                frags = [k for k in pos if re.search(r"_mon[ABC]$", k)]
            elif S in ("S66Rel7",):
                cplx, frags = pos[0], []
            else:
                cplx, frags = pos[0], neg
            e["complex"], e["fragments"] = cplx, frags
            e["shape"] = ("relative energy on a PEC (E(x) - E(R_e)), no monomer terms" if S == "S66Rel7" else
                          "3-body non-additive: trimer + monomers - dimers" if S == "3BHET" else
                          "E(AB) - E(A) - E(B)")
        man["subsets"][S] = dict(category=classify_set(S), description=info.get("Description"),
                                 theory=info.get("Theory level"), n_entries=len(entries),
                                 n_molecules=len(keys), rms_kcal=info.get("RMS energy(kcal/mol)"),
                                 entries=entries,
                                 repo_wb97mv_mae_kcal=float(mae[S][wcol]) if S in mae else None)
        for k in keys:
            if k in man["molecules"]:
                man["molecules"][k]["sets"].append(S); continue
            Z, R = parse_mol(mols[k])
            rec = mols[k]
            m = dict(key=k, sets=[S], n_atoms=len(Z), Z=Z.tolist(), R=R.round(10).tolist(), charge=rec["charge"],
                     mult=rec["multiplicity"], basis_qchem=rec.get("basis"), nbf_qzvppd=rec.get("num_basis"),
                     elements=sorted({SYM[z] for z in Z}, key=lambda s: ZOF[s]),
                     nbf_tzvpd=int(sum(nbf_el.get(SYM[z], np.nan) for z in Z)) if nbf_el and all(
                         SYM[z] in nbf_el for z in Z) else None)
            lab, nc = comps(Z, R)
            m["n_covalent_components"] = int(nc)
            m["cluster_min_distance"] = min_frag_distance(R, lab)
            if cross_check_qchem:
                p = os.path.join(repo, "qchem_inputs", k + ".in")
                if os.path.exists(p):
                    qq = read_qchem_in(p)
                    if qq is None:
                        m["qchem_check"] = "no $molecule block"
                    else:
                        q, mu, Zq, Rq = qq
                        ok = q == m["charge"] and mu == m["mult"] and len(Zq) == len(Z) and np.all(Zq == Z) and \
                            np.abs(Rq - R).max() < 1e-6
                        m["qchem_check"] = "ok" if ok else f"MISMATCH q={q} mult={mu}"
                        m["qchem_sha256"] = sha256(p)
                else:
                    m["qchem_check"] = "no .in file"
            man["molecules"][k] = m
        # fragment maps on complexes
        for e in entries:
            c = man["molecules"][e["complex"]]
            Zc, Rc = np.array(c["Z"]), np.array(c["R"])
            if S == "S66Rel7" or not e["fragments"]:
                lab, nc = comps(Zc, Rc)
                e["fragment_source"] = "cluster split (no monomer terms in the reaction)"
                e["fragment_map"] = lab.tolist()
                e["fragment_charges"] = None if nc != 2 else "neutral complex; per-fragment charges not published"
                e["d_min_source"] = None
                e["d_min_cluster"] = min_frag_distance(Rc, lab)
                e["n_cluster_components"] = int(nc)
                continue
            used, fmap, how = set(), -np.ones(len(Zc), int), []
            for i, fk in enumerate(e["fragments"]):
                f = man["molecules"][fk]
                idx, meth, dev = match_fragment(Zc, Rc, np.array(f["Z"]), np.array(f["R"]), used)
                how.append(dict(fragment=fk, method=meth, dev=dev))
                if idx is not None:
                    fmap[idx] = i; used |= set(idx)
            e["fragment_source"] = "reaction monomer terms"
            e["fragment_match"] = how
            e["fragment_map"] = fmap.tolist()
            e["fragment_charges"] = [man["molecules"][fk]["charge"] for fk in e["fragments"]]
            e["fragment_mults"] = [man["molecules"][fk]["mult"] for fk in e["fragments"]]
            e["fragments_frozen"] = all(h["method"] == "coordinates" for h in how)
            e["fragments_rigid"] = all(h["dev"] is not None and h["dev"] < 1e-3 for h in how)
            if (fmap < 0).any():
                man["problems"].append(f"{e['id']}: {int((fmap < 0).sum())} complex atoms not mapped to a fragment")
            e["d_min_source"] = min_frag_distance(Rc, fmap) if (fmap >= 0).all() else None
            lab, nc = comps(Zc, Rc)
            e["n_cluster_components"] = int(nc)
            e["d_min_cluster"] = min_frag_distance(Rc, lab)
            # does our split agree with the source fragments?
            agree = (fmap >= 0).all() and nc == len(e["fragments"]) and all(
                len(set(lab[fmap == i].tolist())) == 1 for i in range(len(e["fragments"])))
            e["cluster_split_agrees"] = bool(agree)
            qs = e["fragment_charges"]
            if S in ("IHB100x2", "IHB100") or S not in CATEGORY:
                nz = [q for q in qs if q != 0]
                e["category"] = ("ion pair" if len(nz) >= 2 and min(nz) < 0 < max(nz) else
                                 "anion-neutral" if nz and all(q < 0 for q in nz) and len(nz) == 1 else
                                 "cation-neutral" if nz and all(q > 0 for q in nz) and len(nz) == 1 else
                                 "neutral" if not nz else "other charged")
            e["open_shell"] = any(m != 1 for m in e["fragment_mults"]) or c["mult"] != 1
            e["atomic_ion_fragment"] = any(man["molecules"][fk]["n_atoms"] == 1 for fk in e["fragments"])
        for e in entries:
            e.setdefault("category", CATEGORY.get(S, "other"))
            e.setdefault("open_shell", man["molecules"][e["complex"]]["mult"] != 1)
            e.setdefault("atomic_ion_fragment", False)
            # repo wB97M-V check
        errs = [abs(e["wb97mv_qzvppd_ha"] - e["ref_ha"]) * HA_KCAL for e in entries if e["wb97mv_qzvppd_ha"] is not None]
        man["subsets"][S]["wb97mv_qzvppd_mae_kcal_recomputed"] = float(np.mean(errs)) if errs else None
        for e in entries:
            if not (e["composition_balanced"] and e["charge_balanced"]):
                man["problems"].append(f"{e['id']}: composition/charge not balanced")
    # curve grouping (bootstrap unit)
    for S, d in man["subsets"].items():
        for e in d["entries"]:
            e["group"] = curve_group(S, e)
            e["source_coordinate"] = source_coordinate(S, e)
    return man


def curve_group(S, e):
    """Bootstrap unit: a chemical system / curve; every point of a curve stays together."""
    k = e["complex"]
    if S == "S66Rel7":
        return "S66:" + re.match(r"\d+_(\d+)", k).group(1)
    if S in ("IHB100x2", "IHB100"):
        return "IHB:" + re.sub(r"_(080|100|150)$", "", k).replace("IHB100x10_", "")
    if S == "NBC10":
        return "NBC10:" + re.match(r"\d+_(.+)-\d+p\d+_dim", k).group(1)
    if S == "BzDC215":
        return "BzDC:" + re.match(r"\d+_(.+)_\d+p\d+_BzDC215", k).group(1)
    if S == "RG10N":
        return "RG10N:" + re.match(r"RG10N_([A-Za-z]+)_", k).group(1)
    if S == "3BHET":
        return "3BHET:" + re.match(r"3BHET_(\d+)\.", k).group(1)
    return f"{S}:{e['id']}"


def source_coordinate(S, e):
    """The published scan coordinate as named by the source: an R_e scale factor or a separation in Angstrom."""
    k = e["complex"]
    m = re.search(r"(\d+)p(\d+)(?:_dim|_BzDC215|$)", k)
    if S in ("NBC10", "BzDC215", "RG10N") and m:
        return dict(value=float(f"{m.group(1)}.{m.group(2)}"),
                    unit="A (source definition, e.g. centroid separation; not a closest-atom distance)")
    if S in ("IHB100x2", "IHB100"):
        return dict(value=int(k[-3:]) / 100, unit="x R_e (source H-bond distance scaling)")
    if S == "S66Rel7":
        return dict(value=int(re.search(r"(\d{3})_dim", k).group(1)) / 100, unit="x R_e (S66x8 scaling)")
    return None


# --------------------------------------------------------------------------------------------------- exports
def selection(man, sel: str):
    """SELECTION: comma list of SET or SET:entry_id or SET@group (e.g. S66Rel7@S66:01)."""
    ids = []
    for tok in sel.split(","):
        if "@" in tok:
            S, g = tok.split("@", 1)
            ids += [e for e in man["subsets"][S]["entries"] if e["group"] == g]
        elif ":" in tok:
            S, i = tok.split(":", 1)
            ids += [e for e in man["subsets"][S]["entries"] if e["id"] == i]
        else:
            ids += man["subsets"][tok]["entries"]
    return ids


def molecules_for(man, entries, extra_monomers=True):
    keys = []
    for e in entries:
        for _, k in e["terms"]:
            if k not in keys:
                keys.append(k)
        for k in e.get("extra_monomers", []) if extra_monomers else []:
            if k not in keys:
                keys.append(k)
    return keys


def write_xyz(man, keys, path):
    with open(path, "w") as f:
        for k in keys:
            m = man["molecules"][k]
            f.write(f"{m['n_atoms']}\npbc=\"F F F\" Properties=species:S:1:pos:R:3 charge={m['charge']} "
                    f"spin_multiplicity={m['mult']} key={k}\n")
            for z, r in zip(m["Z"], m["R"]):
                f.write(f"{SYM[z]} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n")


def write_points(man, keys, path):
    with open(path, "w") as f:
        for k in keys:
            m = man["molecules"][k]
            f.write(json.dumps(dict(key=_safe(k), gscdb_key=k, Z=m["Z"], R=m["R"], charge=m["charge"], mult=m["mult"],
                                    nbf_tzvpd=m["nbf_tzvpd"])) + "\n")


def _safe(k):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", k)


# ---------------------------------------------------------------------------------------------------- markdown
def manifest_md(man) -> str:
    L = ["# GSCDB137 external validation: manifest", "",
         f"Source: GSCDB137 (Liang & Head-Gordon, arXiv:{man['source']['arxiv']}; doi {man['source']['doi']}), "
         f"{man['source']['repo']} at commit `{man['source']['commit']}` ({man['source']['commit_date']}). "
         f"{man['source']['note']}. Checksums (sha256) of the files read are in manifest.json.", "",
         "Reference values are Hartree in `Info/DatasetEval.csv`; tables here are meV (1 kcal/mol = 43.36 meV).", ""]
    L += ["## Subsets", "",
          "| subset | category | entries | molecules | complexes | monomers | TZVPD nbf (median / max) | "
          "fragments | d_min source (min–max, Å) | d_min cluster split (min–max, Å) | split agrees | "
          "repo ωB97M-V/QZVPPD MAE vs ref (kcal/mol; recomputed) |", "|" + "---|" * 12]
    for S, d in man["subsets"].items():
        ent = d["entries"]
        cx = sorted({e["complex"] for e in ent})
        mo = sorted({k for e in ent for k in e["fragments"]})
        allk = sorted({k for e in ent for _, k in e["terms"]})
        nb = [man["molecules"][k]["nbf_tzvpd"] for k in allk if man["molecules"][k]["nbf_tzvpd"]]
        ds = [e["d_min_source"] for e in ent if e.get("d_min_source") is not None]
        dc = [e["d_min_cluster"] for e in ent if e.get("d_min_cluster") is not None]
        how = Counter(h["method"] for e in ent for h in e.get("fragment_match", []))
        ag = [e.get("cluster_split_agrees") for e in ent if "cluster_split_agrees" in e]
        f = lambda v: f"{min(v):.2f}–{max(v):.2f}" if v else "–"
        L.append(f"| {S} | {d['category']} | {d['n_entries']} | {d['n_molecules']} | {len(cx)} | {len(mo)} | "
                 f"{int(np.median(nb)) if nb else '–'} / {max(nb) if nb else '–'} | "
                 f"{', '.join(f'{k} {v}' for k, v in how.items()) or 'cluster split'} | {f(ds)} | {f(dc)} | "
                 f"{sum(ag)}/{len(ag) if ag else 0} | {d['repo_wb97mv_mae_kcal']:.3f} ({d['wb97mv_qzvppd_mae_kcal_recomputed']:.3f}) |")
    L += ["", "## Problems found", ""] + ([f"- {p}" for p in man["problems"]] or ["- none"])
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------- scoring
def xyz_keys(path):
    """`key=` of every frame header of an export (frame order = prediction order)."""
    L = open(path).read().split("\n"); out, i = [], 0
    while i < len(L) and L[i].strip():
        n = int(L[i]); out.append(re.search(r"key=(\S+)", L[i + 1]).group(1)); i += n + 2
    return out


def load_preds(path, keys):
    J = json.load(open(path))
    fr = J["frames"]
    if len(fr) != len(keys):
        raise ValueError(f"{path}: {len(fr)} frames for {len(keys)} keys")
    E = {k: (None if f.get("energy") is None or "error" in f else float(f["energy"])) for k, f in zip(keys, fr)}
    return E, {k: v for k, v in J.items() if k != "frames"}


def load_dft(path):
    """parse_pilot.py jsonl -> {gscdb key: E [eV]} for converged points (key -> gscdb_key via meta)."""
    out, qc = {}, {}
    for line in open(path):
        r = json.loads(line)
        k = (r.get("meta") or {}).get("gscdb_key", r["key"])
        qc[k] = dict(status=r.get("status"), n_basis=r.get("n_basis"), core_h=r.get("core_h"),
                     n_iter=(r.get("n_iter") or [None])[-1], s2_dev=r.get("s2_dev"))
        if r.get("ok") and r.get("conv") and r.get("E") is not None:
            out[k] = float(r["E"])
    return out, qc


def boot(x, groups, f=np.mean, nb=4000, seed=0):
    """Percentile CI, resampling whole groups (curves / chemical systems); (nan, nan) for one group."""
    x, g = np.asarray(x, float), np.asarray(groups)
    G = np.unique(g)
    if len(G) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed); idx = {k: np.where(g == k)[0] for k in G}
    v = [f(x[np.concatenate([idx[k] for k in rng.choice(G, len(G))])]) for _ in range(nb)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def stat(x, groups):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(n=0, n_groups=0)
    return dict(n=int(len(x)), n_groups=int(len(set(groups))), mae=float(np.mean(np.abs(x))),
                mae_ci=boot(np.abs(x), groups), mse=float(np.mean(x)), mse_ci=boot(x, groups),
                maxabs=float(np.max(np.abs(x))))


def score(man, entries, preds: dict, dft: dict, baseline: str = "PET-OMol-S", aside=(), seen=None):
    """preds: {model: {gscdb key: E eV or None}} (ensembles already averaged). Returns per-entry rows and stats.
    surrogate = dE_model - dE_wB97M-V/TZVPD; end_to_end = dE_model - ref; functional = dE_wB97M-V/TZVPD - ref;
    basis = dE_wB97M-V/TZVPD - dE_wB97M-V/QZVPPD (repo). All meV."""
    rows = []
    for e in entries:
        ks = [k for _, k in e["terms"]]
        comb = lambda E: (sum(c * E[k] for c, k in e["terms"]) * 1000) if all(E.get(k) is not None for k in ks) else None
        d = comb(dft)
        r = dict(id=e["id"], set=e["set"], group=e["group"], category=e["category"], open_shell=e["open_shell"],
                 atomic_ion_fragment=e["atomic_ion_fragment"], complex=e["complex"],
                 source_coordinate=(e.get("source_coordinate") or {}).get("value"),
                 d_min_source=e.get("d_min_source"), d_min_cluster=e.get("d_min_cluster"),
                 seen_by=sorted({h + ("" if k == e["complex"] else ":term") for _, k in e["terms"]
                                 for h in (seen or {}).get(k, [])}),
                 ref=e["ref_meV"], dft=d, functional=None if d is None else d - e["ref_meV"],
                 basis=None if d is None or e.get("wb97mv_qzvppd_ha") is None else d - e["wb97mv_qzvppd_ha"] * HA_EV * 1000,
                 functional_qzvppd=None if e.get("wb97mv_qzvppd_ha") is None else (e["wb97mv_qzvppd_ha"] - e["ref_ha"]) * HA_EV * 1000,
                 model={})
        for m, E in preds.items():
            v = comb(E)
            r["model"][m] = dict(dE=v, surrogate=None if v is None or d is None else v - d,
                                 end_to_end=None if v is None else v - e["ref_meV"])
        rows.append(r)
    models = list(preds)
    # Populations. end-to-end: every core model predicts the entry. surrogate: additionally the ωB97M-V/TZVPD label
    # exists. `aside` models (oracles needing DFT inputs) never restrict a population; they are scored on their own
    # entries and reported with their own n.
    core = [m for m in models if m not in aside]
    pop_e = [r for r in rows if all(r["model"][m]["dE"] is not None for m in core)]
    pop_s = [r for r in pop_e if r["dft"] is not None]
    out = dict(n_entries=len(rows), n_common=len(pop_s), n_common_end_to_end=len(pop_e), models=models,
               aside=list(aside), strata={})
    strata = OrderedDict([("all", lambda r: True)])
    for c in sorted({r["category"] for r in rows}):
        strata[c] = (lambda c: lambda r: r["category"] == c)(c)
    for s_ in sorted({r["set"] for r in rows}):
        strata["set:" + s_] = (lambda s_: lambda r: r["set"] == s_)(s_)
    strata["atomic-ion fragment"] = lambda r: r["atomic_ion_fragment"]
    strata["molecular fragments only"] = lambda r: not r["atomic_ion_fragment"]
    strata["open-shell"] = lambda r: r["open_shell"]
    dist = lambda r: r["d_min_source"] if r["d_min_source"] is not None else r["d_min_cluster"]
    for lo, hi in ((0, 2.5), (2.5, 4), (4, 8), (8, 99)):
        strata[f"closest contact {lo}-{hi} A"] = (lambda lo, hi: lambda r: dist(r) is not None and lo <= dist(r) < hi)(lo, hi)
    if any(r["set"] == "IHB100x2" for r in rows):
        for x in (0.8, 1.5):
            strata[f"IHB100x2 {x} R_e"] = (lambda x: lambda r: r["set"] == "IHB100x2" and r["source_coordinate"] == x)(x)
    strata["unseen (complex not in any store)"] = lambda r: not any(not h.endswith(":term") for h in r.get("seen_by", []))
    for name, f in strata.items():
        st = {}
        for tgt, pop in (("surrogate", pop_s), ("end_to_end", pop_e)):
            R = [r for r in pop if f(r)]
            if tgt == "surrogate":
                g = [r["group"] for r in R]
                st["functional"] = stat([r["functional"] for r in R], g)
                Rb = [r for r in R if r["basis"] is not None]
                st["basis"] = stat([r["basis"] for r in Rb], [r["group"] for r in Rb])
            if tgt == "end_to_end":   # the repository's ωB97M-V/def2-QZVPPD − GSCDB: functional floor on every entry
                Rq = [r for r in R if r["functional_qzvppd"] is not None]
                st["functional_qzvppd"] = stat([r["functional_qzvppd"] for r in Rq], [r["group"] for r in Rq])
            for m in models:
                Rm = [r for r in R if r["model"][m]["dE"] is not None]
                st.setdefault(m, {})[tgt] = stat([r["model"][m][tgt] for r in Rm], [r["group"] for r in Rm])
                if m != baseline and baseline in models and Rm:
                    dd = [abs(r["model"][m][tgt]) - abs(r["model"][baseline][tgt]) for r in Rm]
                    st[m][f"paired_dabs_{tgt}_vs_{baseline}"] = dict(mean=float(np.mean(dd)),
                                                                   ci=boot(dd, [r["group"] for r in Rm]))
                    if tgt == "surrogate":   # old key kept for the report
                        st[m]["paired_dabs_surrogate_vs_" + baseline] = st[m][f"paired_dabs_{tgt}_vs_{baseline}"]
        out["strata"][name] = st
    return rows, out


def curve_stats(rows, models):
    """PEC metrics per curve, x = recomputed closest-contact distance (Å; the source coordinate kept alongside):
    signed error per point, curve MAE, error of the longest increment (the two largest-distance points).
    S66Rel7 rows are E(x) - E(1.00 R_e); NBC10 rows are interaction energies with curve-shared rigid monomers.
    Targets without a label are None."""
    out = {}
    pec = [r for r in rows if r["set"] in ("S66Rel7", "NBC10", "BzDC215") and r["source_coordinate"] is not None]
    for g in sorted({r["group"] for r in pec}):
        dist = lambda r: r["d_min_source"] if r["d_min_source"] is not None else r["d_min_cluster"]
        R = sorted([r for r in pec if r["group"] == g], key=dist)
        if len(R) < 2:
            continue
        o = dict(set=R[0]["set"], d_min=[dist(r) for r in R], source_x=[r["source_coordinate"] for r in R],
                 ref=[r["ref"] for r in R], dft=[r["dft"] for r in R], models={})
        inc = lambda a: None if a[-1] is None or a[-2] is None else a[-1] - a[-2]
        mae = lambda a: None if any(v is None for v in a) else float(np.mean(np.abs(a)))
        for m in models:
            if any(r["model"][m]["dE"] is None for r in R):
                continue
            v = {t: [r["model"][m][t] for r in R] for t in ("surrogate", "end_to_end")}
            o["models"][m] = dict(**v, **{f"curve_mae_{t}": mae(v[t]) for t in v},
                                  **{f"longest_increment_{t}": inc(v[t]) for t in v})
        out[g] = o
    return out


def curve_summary(curves, models):
    """Across curves (one value per curve, so the bootstrap unit is the curve): curve MAE and |longest increment|."""
    out = {}
    for S in sorted({c["set"] for c in curves.values()}):
        C = [c for c in curves.values() if c["set"] == S]
        o = {}
        for m in models:
            for t in ("surrogate", "end_to_end"):
                a = [c["models"][m][f"curve_mae_{t}"] for c in C if m in c["models"] and c["models"][m][f"curve_mae_{t}"] is not None]
                b = [abs(c["models"][m][f"longest_increment_{t}"]) for c in C if m in c["models"] and c["models"][m][f"longest_increment_{t}"] is not None]
                o.setdefault(m, {})[t] = dict(n_curves=len(a), curve_mae=float(np.mean(a)) if a else None,
                                              curve_mae_ci=boot(a, list(range(len(a)))) if len(a) > 1 else None,
                                              abs_longest_increment=float(np.mean(b)) if b else None,
                                              abs_longest_increment_ci=boot(b, list(range(len(b)))) if len(b) > 1 else None)
        out[S] = o
    return out


def oracle_preds(man, entries, atoms, keys):
    """Pseudo-energies (eV) whose reaction combination is the point-charge Coulomb energy between the DFT monomer
    charges (MBIS) of the fragments, placed at the monomers' published coordinates: E(complex) = E_es, E(A) = E(B) = 0.
    Only 2-fragment E(AB) - E(A) - E(B) entries; everything else None (scored on its own population)."""
    out = {k: None for k in keys}
    for e in entries:
        if len(e.get("fragments") or []) != 2 or e["set"] == "3BHET" or not e.get("fragments_frozen"):
            continue  # displaced monomers (IHB100x2) would need the complex-frame placement
        q = [atoms.get(f"{_safe(k)}/mbis_q") if f"{_safe(k)}/mbis_q" in atoms.files else None for k in e["fragments"]]
        if any(x is None for x in q):
            continue
        RA, RB = (np.array(man["molecules"][k]["R"]) for k in e["fragments"])
        r = np.linalg.norm(RA[:, None] - RB[None], axis=-1)
        out[e["complex"]] = float(14.3996454784 * np.sum(np.outer(q[0], q[1]) / r))
        for k in e["fragments"]:
            out[k] = 0.0
    return out


def coverage(man, meta: dict):
    """Per model x subset: entries whose every molecule has covered elements, |charge| <= max and multiplicity <= max.
    meta: {model: {elements: [Z], max_charge, max_mult, source}} from the model files (GGUF composition_keys = elements
    with a trained energy offset; MACE z_table)."""
    out = {}
    for m, d in meta.items():
        el = set(d["elements"]); row = {}
        for S, sd in man["subsets"].items():
            ok, why = 0, Counter()
            for e in sd["entries"]:
                bad = []
                for _, k in e["terms"]:
                    mo = man["molecules"][k]
                    miss = sorted({SYM[z] for z in mo["Z"]} - {SYM[z] for z in el})
                    if miss:
                        bad.append("elements " + "/".join(miss))
                    if abs(mo["charge"]) > d.get("max_charge", 99):
                        bad.append(f"charge {mo['charge']}")
                    if mo["mult"] > d.get("max_mult", 99):
                        bad.append(f"mult {mo['mult']}")
                ok += not bad; why.update(set(bad))
            row[S] = dict(covered=ok, n=len(sd["entries"]), reasons=dict(why))
        out[m] = dict(meta={k: v for k, v in d.items() if k != "elements"},
                      elements=[SYM[z] for z in sorted(el)], subsets=row)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="lrbench.gscdb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest"); p.add_argument("repo"); p.add_argument("outdir")
    p.add_argument("--subsets", default=",".join(P0 + ["NBC10", "BzDC215", "RG10N"]))
    p.add_argument("--bas", default=os.environ.get("LRBENCH_TZVPD_BAS", ""),
                   help="ORCA-printed def2-TZVPD basis file; without it the manifest has no basis-function counts")
    p = sub.add_parser("export"); p.add_argument("manifest"); p.add_argument("selection"); p.add_argument("out")
    p = sub.add_parser("points"); p.add_argument("manifest"); p.add_argument("selection"); p.add_argument("out")
    p = sub.add_parser("coverage"); p.add_argument("manifest"); p.add_argument("meta"); p.add_argument("out")
    p = sub.add_parser("score"); p.add_argument("manifest"); p.add_argument("selection"); p.add_argument("xyz")
    p.add_argument("dft"); p.add_argument("out")
    p.add_argument("--model", action="append", default=[], help="NAME=pred.json[,pred2.json...] (several = "
                   "prediction-averaged ensemble; members also scored as NAME/s<i>)")
    p.add_argument("--baseline", default="PET-OMol-S")
    p.add_argument("--overlap", help="seen.json {molecule: [corpus labels]} from the exact geometry-key store check -> seen_by per entry "
                   "(':term' = only a monomer/dimer/reference term hit)")
    p.add_argument("--oracle", help="parse_pilot *_atoms.npz with <key>/mbis_q of the monomers -> 'oracle' row")
    a = ap.parse_args(argv)
    if a.cmd == "coverage":
        man = json.load(open(a.manifest))
        json.dump(coverage(man, json.load(open(a.meta))), open(a.out, "w"), indent=1); print(a.out); return
    if a.cmd == "score":
        man = json.load(open(a.manifest)); ent = selection(man, a.selection); keys = xyz_keys(a.xyz)
        dft, qc = {}, {}
        for pth in ([] if a.dft == "none" else a.dft.split(",")):
            d_, q_ = load_dft(pth); dft.update(d_); qc.update(q_)
        preds, info = OrderedDict(), {}
        for spec in a.model:
            name, paths = spec.split("=", 1); paths = paths.split(",")
            mem = [load_preds(p, keys) for p in paths]
            info[name] = dict(files=paths, headers=[h for _, h in mem])
            preds[name] = {k: (None if any(E[k] is None for E, _ in mem) else float(np.mean([E[k] for E, _ in mem])))
                           for k in keys}
            if len(mem) > 1:
                info[name]["member_names"] = [f"{name}/m{i}" for i in range(len(mem))]
        if a.oracle:
            preds["oracle"] = oracle_preds(man, ent, np.load(a.oracle), keys)
            info["oracle"] = dict(files=[a.oracle], what="point Coulomb between DFT (ORCA MBIS) monomer charges at the "
                                  "complex geometry; E(monomer) = 0; needs DFT inputs")
        seen = json.load(open(a.overlap)) if a.overlap else None   # {molecule key: [corpus labels]}
        rows, st = score(man, ent, preds, dft, a.baseline, aside=("oracle",) if a.oracle else (), seen=seen)
        members = OrderedDict()
        for spec in a.model:
            name, paths = spec.split("=", 1); paths = paths.split(",")
            if len(paths) > 1:
                for i, p in enumerate(paths):
                    members[f"{name}/m{i}"] = load_preds(p, keys)[0]
        mrows, mst = score(man, ent, members, dft, baseline="__none__") if members else ([], {})
        out = dict(selection=a.selection, xyz=a.xyz, dft=a.dft, dft_qc=qc, model_info=info, summary=st,
                   curves=(cv := curve_stats(rows, list(preds))), curve_summary=curve_summary(cv, list(preds)), rows=rows, members_summary=mst.get("strata") if mst else None)
        json.dump(out, open(a.out, "w"), indent=1, default=float); print(a.out, "surrogate", st["n_common"], "end-to-end", st["n_common_end_to_end"], "of", st["n_entries"]); return
    if a.cmd == "manifest":
        os.makedirs(a.outdir, exist_ok=True)
        man = build_manifest(a.repo, a.subsets.split(","), a.bas)
        json.dump(man, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=1)
        print(manifest_md(man))
    else:
        man = json.load(open(a.manifest))
        keys = molecules_for(man, selection(man, a.selection))
        (write_xyz if a.cmd == "export" else write_points)(man, keys, a.out)
        print(a.out, len(keys), "molecules")


if __name__ == "__main__":
    main()
