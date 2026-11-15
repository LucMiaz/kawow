"""Apply tests/test_data/naef2024_corrections.csv to the Naef & Acree (2024)
supplementary SDF files S01-S03 in tests/test_data/.

The supplementary data of Naef & Acree (Liquids 2024, 4, 231-260) contain a few
records whose structure does not match the compound name (the value is the one
of the named compound), and a few implausible values. The corrections file
lists them: ``replace_structure`` swaps the molecule block for the structure of
the named compound and keeps the record's properties; ``remove`` deletes the
record. All other records are copied byte for byte.

The script is idempotent: a record that already has the corrected structure,
or a removed record, is left as it is. It stops if a record has neither the
expected old structure nor the corrected one, so it can only be applied to the
published files or to files it produced itself.

Usage (from the repository root)::

    python scripts/correct_naef_data.py            # correct the files in place
    python scripts/correct_naef_data.py --check    # report only
"""

import argparse
import csv
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "tests" / "test_data"
CORRECTIONS = DATA / "naef2024_corrections.csv"
FILES = {
    "S01": DATA / "S01. Compounds List for logPow-Parameters Calculations.sdf",
    "S02": DATA / "S02. Compounds List for logKoa-Parameters Calculations.sdf",
    "S03": DATA / "S03. Compounds List with exp logKaw Data.sdf",
}

_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def _skeleton(mol):
    """First block of the parent's InChIKey: connectivity, without stereo or charge."""
    mol = _uncharger.uncharge(_chooser.choose(mol))
    return Chem.MolToInchiKey(mol)[:14]


def _split_records(text):
    """Split SDF text into records, each ending with its '$$$$' line."""
    parts = text.split("$$$$")
    records = [p + "$$$$" for p in parts[:-1]]
    tail = parts[-1]
    return records, tail


def _alias_name(record):
    lines = record.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(">") and "<Alias name>" in line and i + 1 < len(lines):
            return lines[i + 1].strip()
    return None


def _molblock_from_smiles(smiles, title):
    """3D molecule block with explicit hydrogens, as in the published files."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        AllChem.Compute2DCoords(mol)
    else:
        AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp("_Name", title)
    return Chem.MolToMolBlock(mol)


def correct_file(key, corrections, check_only):
    path = FILES[key]
    with open(path, encoding="latin-1", newline="") as fh:  # keep line endings as published
        text = fh.read()
    records, tail = _split_records(text)
    by_name = {}
    for i, rec in enumerate(records):
        by_name.setdefault(_alias_name(rec), []).append(i)

    to_remove, done = set(), []
    for c in corrections:
        idx = by_name.get(c["name"], [])
        if c["action"] == "remove":
            if not idx:
                done.append(f"  {c['name']}: already removed")
                continue
            if len(idx) != 1:
                raise SystemExit(f"{key}: {len(idx)} records named {c['name']!r}")
            to_remove.add(idx[0])
            done.append(f"  {c['name']}: removed (value {c['value']})")
            continue
        if len(idx) != 1:
            raise SystemExit(f"{key}: {len(idx)} records named {c['name']!r}")
        i = idx[0]
        rec = records[i]
        start = rec.find("\n>")  # end of the molecule block
        molblock, props = rec[:start], rec[start:]
        mol = Chem.MolFromMolBlock(molblock.lstrip("\n"), sanitize=True, removeHs=False)
        current = _skeleton(mol)
        new = _skeleton(Chem.MolFromSmiles(c["new_smiles"]))
        old = _skeleton(Chem.MolFromSmiles(c["old_smiles"]))
        if current == new:
            done.append(f"  {c['name']}: already corrected")
            continue
        if current != old:
            raise SystemExit(f"{key}: {c['name']!r} has neither the published nor the corrected structure")
        lead = molblock[: len(molblock) - len(molblock.lstrip("\n"))]
        title = molblock.lstrip("\n").splitlines()[0]
        new_block = _molblock_from_smiles(c["new_smiles"], title).rstrip("\n")
        records[i] = lead + new_block + props
        done.append(f"  {c['name']}: structure replaced")

    print(f"{path.name}:")
    print("\n".join(done))
    if not check_only:
        kept = [r for i, r in enumerate(records) if i not in to_remove]
        with open(path, "w", encoding="latin-1", newline="") as fh:
            fh.write("".join(kept) + tail)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    args = parser.parse_args(argv)
    with open(CORRECTIONS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for key in FILES:
        correct_file(key, [r for r in rows if r["file"] == key], args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
