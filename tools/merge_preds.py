"""merge predictions of periodic_box_open + periodic_box_pbc back into periodic_box order: merge_preds.py FULL.extxyz OPEN.json PBC.json OUT.json"""
import json, sys
from ase.io import read
A = read(sys.argv[1], ":"); o = json.load(open(sys.argv[2]))["frames"]; p = json.load(open(sys.argv[3]))["frames"]
io = ip = 0; F = []
for a in A:
    if a.info["L"] == 0: F.append(o[io]); io += 1
    else: F.append(p[ip]); ip += 1
json.dump({"frames": F, "merged_from": sys.argv[2:4]}, open(sys.argv[4], "w"))
