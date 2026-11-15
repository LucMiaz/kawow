"""Check the structures of the Naef & Acree (2024) datasets S01-S03 against PubChem.

Every record's compound name is resolved with PubChemCompounds (PubChem PUG REST:
name -> CID, then the CID's isomeric SMILES and InChIKey), and the structure
returned by PubChem is compared with the structure in the SDF file. When
``tests/test_data/naef2024_corrections.csv`` is available, the published and the
corrected structure are both compared, so the table shows for each correction
whether PubChem agrees with it.

Structures are compared at four levels, from strict to loose:

``identical``         same full InChIKey
``stereo/isotopes``   same InChIKey first block (connectivity), stereo or isotopes differ
``parent``            same first block after keeping the largest fragment and
                      neutralising charges (salts, protonation states)
``tautomer``          same first block after RDKit's canonical tautomer
``different``         none of the above
``not found``         PubChem did not resolve the name

The first CID PubChem returns for a name is used (PubChem orders them by
relevance); ``n_cids`` gives how many CIDs the name matched. Names that match
another compound in PubChem (trade names, abbreviations) appear as
``different``: those need a manual look, see ``reviewed`` in the corrections file.

Usage (from the repository root)::

    python scripts/check_naef_structures_pubchem.py
    python scripts/check_naef_structures_pubchem.py --sdf-dir <dir> --corrections <csv> --out <csv>

The PubChem answers are cached in ``--cache`` (JSON), so an interrupted run
resumes where it stopped. A full run (about 5 000 names) takes 20-40 minutes at
PubChem's rate limit of 5 requests per second.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "tests" / "test_data"
FILES = {
    "S01": ("S01. Compounds List for logPow-Parameters Calculations.sdf", "logP"),
    "S02": ("S02. Compounds List for logKoa-Parameters Calculations.sdf", "logKoa"),
    "S03": ("S03. Compounds List with exp logKaw Data.sdf", "logKaw"),
}
LEVELS = ["identical", "stereo/isotopes", "parent", "tautomer", "different"]

_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()
_tautomers = rdMolStandardize.TautomerEnumerator()


# --------------------------------------------------------------------------- #
# Structure comparison
# --------------------------------------------------------------------------- #
def _keys(mol):
    """InChIKeys of a molecule at the four comparison levels (None if not computable)."""
    if mol is None:
        return None
    try:
        full = Chem.MolToInchiKey(mol)
        parent = _uncharger.uncharge(_chooser.choose(Chem.Mol(mol)))
        parent_key = Chem.MolToInchiKey(parent)
        try:
            taut_key = Chem.MolToInchiKey(_tautomers.Canonicalize(parent))
        except Exception:  # tautomer enumeration can fail on exotic structures
            taut_key = parent_key
    except Exception:
        return None
    if not full:
        return None
    return {"identical": full, "stereo/isotopes": full[:14], "parent": parent_key[:14], "tautomer": taut_key[:14]}


def compare(mol_a, mol_b):
    """Strictest level at which the two structures agree."""
    a, b = _keys(mol_a), _keys(mol_b)
    if a is None or b is None:
        return "not comparable"
    for level in LEVELS[:-1]:
        if a[level] == b[level]:
            return level
    return "different"


def _smiles(mol):
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))
    except Exception:
        return Chem.MolToSmiles(mol) if mol is not None else ""


# --------------------------------------------------------------------------- #
# PubChem (through PubChemCompounds), with a JSON cache
# --------------------------------------------------------------------------- #
class PubChem:
    """Name -> CIDs and CID -> (SMILES, InChIKey), cached on disk."""

    def __init__(self, cache_path, retries=3):
        from pubchem_compounds import get_from_cids, single_synonym_to_pubchem  # noqa: PLC0415

        self._names_to_cids = single_synonym_to_pubchem
        self._get_from_cids = get_from_cids
        self.cache_path = Path(cache_path)
        self.retries = retries
        self.cache = {"names": {}, "cids": {}}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def save(self):
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def resolve(self, names):
        todo = [n for n in dict.fromkeys(names) if n and n not in self.cache["names"]]
        for i, name in enumerate(todo, 1):
            cids = None
            for attempt in range(self.retries):
                # names go into the URL path: encode everything, "/" included
                cids = self._names_to_cids(quote(name, safe=""))
                if cids is not None:
                    break
                time.sleep(1 + attempt)  # not found, or PubChem busy: try again
            self.cache["names"][name] = cids or []
            if i % 100 == 0 or i == len(todo):
                self.save()
                print(f"  names resolved: {i}/{len(todo)}", flush=True)
        return {n: self.cache["names"].get(n, []) for n in names if n}

    def structures(self, cids, batch=100):
        todo = sorted({c for c in cids if str(c) not in self.cache["cids"]})
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            data = None
            for attempt in range(self.retries):
                data = self._get_from_cids(chunk, target="property/IsomericSMILES,InChIKey")
                if data and "PropertyTable" in data:
                    break
                time.sleep(1 + attempt)
            for prop in (data or {}).get("PropertyTable", {}).get("Properties", []):
                smiles = prop.get("SMILES") or prop.get("IsomericSMILES") or prop.get("CanonicalSMILES") or ""
                self.cache["cids"][str(prop["CID"])] = [smiles, prop.get("InChIKey", "")]
            self.save()
            print(f"  structures fetched: {min(start + batch, len(todo))}/{len(todo)}", flush=True)
        return {c: self.cache["cids"].get(str(c)) for c in cids}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def read_records(sdf_dir):
    records = []
    for key, (fname, prop) in FILES.items():
        for i, mol in enumerate(Chem.SDMolSupplier(str(Path(sdf_dir) / fname), sanitize=False, removeHs=False)):
            if mol is None:
                records.append({"file": key, "index": i, "name": "", "value": "", "mol": None})
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            name = mol.GetProp("Alias name").strip() if mol.HasProp("Alias name") else ""
            value = mol.GetProp(prop).strip() if mol.HasProp(prop) else ""
            records.append({"file": key, "index": i, "name": name, "value": value, "mol": mol})
    return records


def read_corrections(path):
    if path is None or not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["file"], r["name"]): r for r in csv.DictReader(fh)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sdf-dir", default=DATA, type=Path, help="folder with the S01-S03 SDF files")
    parser.add_argument("--corrections", default=DATA / "naef2024_corrections.csv", type=Path)
    parser.add_argument("--out", default=DATA / "naef2024_pubchem_check.csv", type=Path)
    parser.add_argument("--cache", default=DATA / "naef2024_pubchem_cache.json", type=Path)
    args = parser.parse_args(argv)

    records = read_records(args.sdf_dir)
    corrections = read_corrections(args.corrections)
    print(f"{len(records)} records, {len(corrections)} corrections")

    pubchem = PubChem(args.cache)
    names = [r["name"] for r in records]
    cids_by_name = pubchem.resolve(names)
    first_cids = [c[0] for c in cids_by_name.values() if c]
    structures = pubchem.structures(first_cids)

    rows = []
    for r in records:
        corr = corrections.get((r["file"], r["name"]), {})
        action = corr.get("action", "")
        current = r["mol"]
        published, corrected = current, None
        if action == "replace_structure":
            old_mol = Chem.MolFromSmiles(corr["old_smiles"])
            new_mol = Chem.MolFromSmiles(corr["new_smiles"])
            # the files may be the published ones or the corrected ones
            if compare(current, new_mol) in LEVELS[:3]:
                published, corrected = old_mol, current
            else:
                published, corrected = current, new_mol
        cids = cids_by_name.get(r["name"], [])
        cid = cids[0] if cids else None
        pc = structures.get(cid) if cid else None
        pc_mol = Chem.MolFromSmiles(pc[0]) if pc and pc[0] else None
        rows.append({
            "file": r["file"],
            "index": r["index"],
            "name": r["name"],
            "value": r["value"],
            "correction": action,
            "published_smiles": _smiles(published) if published is not None else "",
            "corrected_smiles": _smiles(corrected) if corrected is not None else "",
            "pubchem_cid": cid or "",
            "n_cids": len(cids),
            "pubchem_smiles": pc[0] if pc else "",
            "pubchem_inchikey": pc[1] if pc else "",
            "match_published": compare(published, pc_mol) if pc_mol is not None else "not found",
            "match_corrected": (compare(corrected, pc_mol) if pc_mol is not None else "not found") if corrected is not None else "",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {args.out}")
    for key in FILES:
        sub = [x for x in rows if x["file"] == key]
        counts = {}
        for x in sub:
            counts[x["match_published"]] = counts.get(x["match_published"], 0) + 1
        print(f"{key}: {len(sub)} records; published structure vs PubChem: "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    fixed = [x for x in rows if x["correction"] == "replace_structure"]
    if fixed:
        agree = sum(x["match_corrected"] in LEVELS[:4] for x in fixed)
        print(f"corrected records: {agree} of {len(fixed)} agree with PubChem after correction "
              f"(published structure agreed for {sum(x['match_published'] in LEVELS[:4] for x in fixed)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
