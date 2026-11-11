"""Convert small, licence-checked subsamples of public long-range sets to chemiscope files that follow PROPERTIES.md.

Sets (downloaded raw files live in explorer/external/dl_*; nothing downloaded is executed):
  sapt10k  SO3LR Tauri Zenodo 10.5281/zenodo.22920714 (CC BY 4.0), test_benchmarks/table2_external/sapt10k.xyz
           (Villot et al. 2024), reference SAPT2+(3)(CCD)/aug-cc-pVTZ interaction energies
  pla15    same record, pla15.xyz (Kriz & Rezac 2020), protein-ligand complexes
  ko4g     Ko et al. 2021 4G-HDNNP data, Materials Cloud 10.24435/materialscloud:f3-yh (record bk6kw-4b895, CC BY 4.0),
           FHI-aims PBE 'light' energies, forces, Hirshfeld charges (RuNNer input.data, atomic units)
  des370k  DES370K (Donchev et al. 2021), Zenodo 10.5281/zenodo.5676266 (CC BY 4.0), CCSD(T)/CBS interaction energies;
           subsample of the separated (long-range) dimers

Usage: <lrbench venv python> convert_external.py [sapt10k] [pla15] [ko4g] [des370k]
Output: explorer/space/data/ext_<name>.json.gz
"""
import sys, os, json, collections
import numpy as np
import ase, ase.io
import chemiscope

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = f"{HERE}/external"
OUT = f"{HERE}/space/data"
KCAL2MEV = 43.364104  # 1 kcal/mol in meV
HA2MEV = 27211.386246
BOHR = 0.529177210903
rng = np.random.default_rng(0)


def nan(x):
    return float("nan") if x is None else float(x)


def ranges(sel):
    out = []
    for part in str(sel).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a) - 1, int(b or a)))
    return out


def min_dist(pos, ia, ib):
    a, b = pos[ia], pos[ib]
    return float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)))


def write(name, title, desc, level, licence, source, atoms, sprops, aprops, settings):
    """sprops: name -> (values, units, description); aprops likewise (per atom, concatenated)."""
    n = len(atoms)
    na = sum(len(a) for a in atoms)
    P = {"entry": {"target": "structure", "values": ["structure"] * n, "description": "one entry = one structure"},
         "level": {"target": "structure", "values": [level] * n, "description": "level of theory of the reference values (never pool energies across levels)"},
         "set": {"target": "structure", "values": [name] * n, "description": "dataset"},
         "n_atoms": {"target": "structure", "values": [float(len(a)) for a in atoms], "units": "", "description": "number of atoms"}}
    for k, (v, u, d) in sprops.items():
        assert len(v) == n, k
        P[k] = {"target": "structure", "values": list(v), "units": u, "description": d}
    for k, (v, u, d) in aprops.items():
        assert len(v) == na, k
        P[k] = {"target": "atom", "values": list(v), "units": u, "description": d}
    meta = {"name": title, "description": f"{desc} Level: {level}. Licence: {licence}. Source: {source}. "
                                          "Subsample converted for the omol-longrange-explorer; property names follow PROPERTIES.md.",
            "references": [source, "Converted with convert_external.py (omol-longrange-explorer)"]}
    env = chemiscope.all_atomic_environments(atoms, cutoff=3.5)
    path = f"{OUT}/ext_{name}.json.gz"
    chemiscope.write_input(path, structures=atoms, metadata=meta, properties=P, environments=env, settings=settings)
    sz = os.path.getsize(path)
    print(f"ext_{name}: {n} structures, {na} atoms, {sz/1e6:.2f} MB")
    return {"file": f"data/ext_{name}.json.gz", "n_entries": n, "n_atoms": na, "size_bytes": sz, "level": level,
            "licence": licence, "source": source, "properties": sorted(P)}


def st(x, y, color, target="structure", pinned=None):
    s = {"target": target, "map": {"x": {"property": x}, "y": {"property": y}, "color": {"property": color, "palette": "tab10"}},
         "structure": [{"atomLabels": False, "unitCell": True, "keepOrientation": False, "environments": {"activated": False}}]}
    if pinned is not None:
        s["pinned"] = [pinned]
    return s


def dimer_pairs_xyz(path):
    """SO3LR table2 dimer files: consecutive dimer_original / dimer_translated frames."""
    frames = ase.io.read(path, index=":")
    return [f for f in frames if f.info.get("structure_type") == "dimer_original"]


def dimer_set(name, frames, level, licence, source, title, desc, cls_fn):
    atoms, sp = [], collections.defaultdict(list)
    fid, qf = [], []
    for f in frames:
        i = f.info
        ia, ib = ranges(i["selection_a"]), ranges(i["selection_b"])
        pos = f.get_positions()
        a = ase.Atoms(symbols=f.get_chemical_symbols(), positions=pos)
        atoms.append(a)
        sp["id"].append(str(i.get("system_id")))
        sp["class"].append(cls_fn(i))
        sp["charge"].append(float(i.get("charge", 0)))
        sp["spin"].append(float(i.get("multiplicity", 1)))
        sp["charge_A"].append(float(i.get("charge_a", 0)))
        sp["charge_B"].append(float(i.get("charge_b", 0)))
        sp["distance"].append(round(min_dist(pos, ia, ib), 4))
        sp["E_int/REF"].append(round(float(i["ref_Eint"]) * KCAL2MEV, 3))
        f_ = np.full(len(a), np.nan); f_[ia] = 0; f_[ib] = 1
        q_ = np.zeros(len(a)); q_[ia[0]] = float(i.get("charge_a", 0)); q_[ib[0]] += float(i.get("charge_b", 0))
        fid += f_.tolist(); qf += q_.tolist()
    sprops = {
        "id": (sp["id"], "", "system id in the source file"),
        "class": (sp["class"], "", "subset / class"),
        "charge": (sp["charge"], "e", "total charge, e"),
        "spin": (sp["spin"], "", "multiplicity 2S+1"),
        "charge_A": (sp["charge_A"], "e", "charge of monomer A, e"),
        "charge_B": (sp["charge_B"], "e", "charge of monomer B, e"),
        "distance": (sp["distance"], "Å", "minimum intermolecular atom-atom distance, Å"),
        "E_int/REF": (sp["E_int/REF"], "meV", f"reference interaction energy ({level}), meV (from kcal/mol)"),
    }
    aprops = {"fragment_id": (fid, "", "monomer label (0 = A, 1 = B) from selection_a/b"),
              "q_formal": (qf, "e", "monomer charge placed on the monomer's first atom (fragment-level formal charge), e")}
    return write(name, title, desc, level, licence, source, atoms, sprops, aprops, st("distance", "E_int/REF", "class"))


def run_sapt10k():
    p = f"{EXT}/dl_so3lr/test_benchmarks/table2_external/sapt10k.xyz"
    fr = dimer_pairs_xyz(p)
    charged = [f for f in fr if f.info.get("charge_a", 0) != 0 or f.info.get("charge_b", 0) != 0]
    neutral = [f for f in fr if f not in charged] if False else [f for f in fr if f.info.get("charge_a", 0) == 0 and f.info.get("charge_b", 0) == 0]
    ci = rng.choice(len(charged), min(400, len(charged)), replace=False)
    ni = rng.choice(len(neutral), min(200, len(neutral)), replace=False)
    sel = [charged[i] for i in sorted(ci)] + [neutral[i] for i in sorted(ni)]
    print(f"sapt10k: {len(fr)} dimers, {len(charged)} with a charged monomer; kept {len(ci)} charged + {len(ni)} neutral")

    def cls(i):
        qa, qb = i.get("charge_a", 0), i.get("charge_b", 0)
        if qa and qb:
            return "ion_pair" if qa * qb < 0 else "same_sign"
        if qa or qb:
            return "ion_neutral"
        return "neutral"
    return dimer_set("sapt10k", sel, "SAPT2+(3)(CCD)/aug-cc-pVTZ", "CC BY 4.0 (SO3LR Tauri Zenodo record 22920714; SAPT10k: Villot et al. 2024)",
                     "https://doi.org/10.5281/zenodo.22920714", "SAPT10k subsample (charged dimers)",
                     f"SAPT10k dimers at their bound geometries: {len(ci)} of the {len(charged)} dimers with a charged monomer and {len(ni)} random neutral ones (seed 0). "
                     "E_int/REF in meV; distance = minimum intermolecular distance (Å).", cls)


def run_pla15():
    p = f"{EXT}/dl_so3lr/test_benchmarks/table2_external/pla15.xyz"
    fr = dimer_pairs_xyz(p)

    def cls(i):
        return "protein_ligand_" + ("charged" if (i.get("charge_a", 0) or i.get("charge_b", 0)) else "neutral")
    return dimer_set("pla15", fr, "MP2-F12 + dDLPNO-CCSD(T) (PLA15 reference)", "CC BY 4.0 (SO3LR Tauri Zenodo record 22920714; PLA15: Kriz & Rezac 2020)",
                     "https://doi.org/10.5281/zenodo.22920714", "PLA15 protein-ligand complexes",
                     "All 15 PLA15 protein-ligand complexes (pocket + ligand). E_int/REF in meV; distance = minimum pocket-ligand distance (Å).", cls)


def read_runner(path, keep):
    """RuNNer input.data reader (atomic units -> Å, meV, eV/Å). keep: set of structure indices to return."""
    out, cur, k = [], None, -1
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "begin":
                k += 1
                cur = {"lat": [], "sym": [], "pos": [], "q": [], "F": [], "E": None, "Q": 0.0, "comment": ""} if k in keep else None
            elif cur is None:
                continue
            elif t[0] == "comment":
                cur["comment"] = line[8:].strip()
            elif t[0] == "lattice":
                cur["lat"].append([float(x) * BOHR for x in t[1:4]])
            elif t[0] == "atom":
                cur["pos"].append([float(x) * BOHR for x in t[1:4]]); cur["sym"].append(t[4]); cur["q"].append(float(t[5]))
                cur["F"].append([float(x) * HA2MEV / 1e3 / BOHR for x in t[7:10]])
            elif t[0] == "energy":
                cur["E"] = float(t[1]) * HA2MEV
            elif t[0] == "charge":
                cur["Q"] = float(t[1])
            elif t[0] == "end":
                cur["idx"] = k; out.append(cur); cur = None
    return out


def count_runner(path):
    return sum(1 for l in open(path) if l.startswith("begin"))


def run_ko4g():
    systems = {"Carbon_chain": 300, "Ag_cluster": 300, "NaCl": 300, "AuMgO": 80}
    atoms, sp, ap = [], collections.defaultdict(list), collections.defaultdict(list)
    for sysname, nkeep in systems.items():
        p = f"{EXT}/dl_ko/datasets/{sysname}/input.data"
        n = count_runner(p)
        keep = set(rng.choice(n, min(nkeep, n), replace=False).tolist())
        recs = read_runner(p, keep)
        Emin = collections.defaultdict(lambda: np.inf)
        for r in recs:
            Emin[(sysname, r["Q"], len(r["sym"]))] = min(Emin[(sysname, r["Q"], len(r["sym"]))], r["E"])
        for r in recs:
            a = ase.Atoms(symbols=r["sym"], positions=r["pos"])
            if len(r["lat"]) == 3:
                a.cell = r["lat"]; a.pbc = True
            atoms.append(a)
            F = np.array(r["F"])
            sp["id"].append(f"{sysname}:{r['idx']}")
            sp["class"].append(f"{sysname} Q={r['Q']:+g}")
            sp["system"].append(sysname)
            sp["charge"].append(r["Q"])
            sp["E_rel/REF"].append(round(r["E"] - Emin[(sysname, r["Q"], len(r["sym"]))], 3))
            sp["F_rms/REF"].append(round(float(np.sqrt((F ** 2).sum(1).mean())), 4))
            pos = np.array(r["pos"])
            sp["distance"].append(round(float(np.linalg.norm(pos[0] - pos[3])), 4) if sysname == "NaCl" else float("nan"))
            ap["q_hirshfeld"] += [round(x, 4) for x in r["q"]]
            ap["F_norm/REF"] += [round(float(x), 4) for x in np.linalg.norm(F, axis=1)]
        print(f"ko4g {sysname}: {n} structures, kept {len(recs)}")
    level = "PBE (FHI-aims, light)"
    sprops = {
        "id": (sp["id"], "", "system:index in input.data"),
        "class": (sp["class"], "", "system and total charge"),
        "system": (sp["system"], "", "Ko et al. system"),
        "charge": (sp["charge"], "e", "total charge, e"),
        "E_rel/REF": (sp["E_rel/REF"], "meV", f"total energy relative to the lowest kept structure of the same system, charge and size ({level}), meV"),
        "F_rms/REF": (sp["F_rms/REF"], "eV/Å", "RMS reference force, eV/Å"),
        "distance": (sp["distance"], "Å", "NaCl only: distance between atom 1 (displaced Na) and atom 4, Å"),
    }
    aprops = {"q_hirshfeld": (ap["q_hirshfeld"], "e", "Hirshfeld charge from the reference calculation, e"),
              "F_norm/REF": (ap["F_norm/REF"], "eV/Å", "|reference force| per atom, eV/Å")}
    return write("ko4g", "Ko et al. 4G-HDNNP systems", "Subsample (seed 0) of the four 4G-HDNNP benchmark systems of Ko et al. 2021: "
                 "C10H2/C10H3+ chains, Ag3+/Ag3- clusters, Na9Cl8+/Na8Cl8 clusters, Au2 on (Al-doped) MgO(001) (periodic). "
                 "These test non-local charge transfer at fixed geometry class. Energies are relative within system/charge.",
                 level, "CC BY 4.0 (Materials Cloud Archive record bk6kw-4b895)", "https://archive.materialscloud.org/records/bk6kw-4b895",
                 atoms, sprops, aprops, st("distance", "E_rel/REF", "class"))


def run_des370k():
    """Charged radial scans (qm_opt_dimer, k_index = 0.1 A offsets) out to the largest separations."""
    import csv
    p = f"{EXT}/dl_des370k/DES370K.csv"
    groups = collections.defaultdict(list)
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r["group_orig"] != "qm_opt_dimer":
                continue
            groups[r["group_id"]].append(r)
    def kind(r):
        qa, qb = int(r["charge0"]), int(r["charge1"])
        if qa and qb:
            return "ion_pair" if qa * qb < 0 else "same_sign"
        return "ion_neutral" if (qa or qb) else "neutral"
    by = collections.defaultdict(list)
    for g, rs in groups.items():
        if len(rs) >= 8:
            by[kind(rs[0])].append(g)
    want = {"ion_pair": 12, "same_sign": 6, "ion_neutral": 12, "neutral": 6}
    sel = []
    for k, n in want.items():
        gs = sorted(by[k])
        if gs:
            sel += [gs[i] for i in sorted(rng.choice(len(gs), min(n, len(gs)), replace=False))]
    atoms, sp, fid, qf = [], collections.defaultdict(list), [], []
    for g in sel:
        rs = sorted(groups[g], key=lambda r: int(r["k_index"]))
        for r in rs:
            el = r["elements"].split()
            xyz = np.array([float(x) for x in r["xyz"].split()]).reshape(-1, 3)
            n0 = int(r["natoms0"])
            a = ase.Atoms(symbols=el, positions=xyz)
            atoms.append(a)
            ia, ib = list(range(n0)), list(range(n0, len(el)))
            sp["id"].append(f"DES370K_{r['geom_id']}")
            sp["group"].append(f"{r['smiles0']} | {r['smiles1']} | scan {g}")
            sp["class"].append(kind(r))
            sp["charge"].append(float(int(r["charge0"]) + int(r["charge1"])))
            sp["charge_A"].append(float(r["charge0"])); sp["charge_B"].append(float(r["charge1"]))
            sp["k_index"].append(float(r["k_index"]))
            sp["distance"].append(round(min_dist(xyz, ia, ib), 4))
            sp["E_int/REF"].append(round(float(r["cbs_CCSD(T)_all"]) * KCAL2MEV, 3))
            sp["E_int_sapt_es/REF"].append(round(float(r["sapt_es"]) * KCAL2MEV, 3) if r["sapt_es"] else float("nan"))
            f_ = np.zeros(len(el)); f_[ib] = 1
            q_ = np.zeros(len(el)); q_[0] = float(r["charge0"]); q_[n0] += float(r["charge1"])
            fid += f_.tolist(); qf += q_.tolist()
    level = "CCSD(T)/CBS (DES370K)"
    print(f"des370k: {len(groups)} qm_opt_dimer scans; kept {len(sel)} scans, {len(atoms)} geometries")
    sprops = {
        "id": (sp["id"], "", "DES370K geom_id"),
        "group": (sp["group"], "", "monomer SMILES and radial-scan group id"),
        "class": (sp["class"], "", "ion_pair / same_sign / ion_neutral / neutral (monomer charges)"),
        "charge": (sp["charge"], "e", "total charge, e"),
        "charge_A": (sp["charge_A"], "e", "monomer 0 charge, e"), "charge_B": (sp["charge_B"], "e", "monomer 1 charge, e"),
        "k_index": (sp["k_index"], "", "radial offset index (0.1 A steps from the optimised dimer)"),
        "distance": (sp["distance"], "Å", "minimum intermolecular atom-atom distance, Å"),
        "E_int/REF": (sp["E_int/REF"], "meV", f"reference interaction energy, cbs_CCSD(T)_all ({level}), meV (from kcal/mol)"),
        "E_int_sapt_es/REF": (sp["E_int_sapt_es/REF"], "meV", "SAPT0 electrostatics component (sapt_es; DES370K, SAPT0 level), meV"),
    }
    aprops = {"fragment_id": (fid, "", "monomer label (0, 1)"), "q_formal": (qf, "e", "monomer charge on the monomer's first atom, e")}
    s_ = st("distance", "E_int/REF", "class")
    s_["map"]["joinPoints"] = False
    return write("des370k", "DES370K radial scans (charged)", f"{len(sel)} DES370K radial scans (qm_opt_dimer groups with >= 8 points; "
                 "12 ion pairs, 6 same-sign, 12 ion-neutral, 6 neutral; seed 0), every point of each scan. E_int/REF in meV; distance in Å. "
                 "DES370K scans reach only a few Å beyond contact, so this probes the onset of the long-range regime.",
                 level, "DESRES Data Sets License (redistribution permitted with the copyright notice; Zenodo record lists CC BY 4.0); "
                 "Copyright 2020, D. E. Shaw Research. All rights reserved.", "https://doi.org/10.5281/zenodo.5676266",
                 atoms, sprops, aprops, s_)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["sapt10k", "pla15", "ko4g", "des370k"]
    man_p = f"{OUT}/manifest_external.json"
    man = json.load(open(man_p)) if os.path.exists(man_p) else {}
    for t in todo:
        r = globals()["run_" + t]()
        if isinstance(r, dict):
            man["ext_" + t] = r
    json.dump(man, open(man_p, "w"), indent=1)
