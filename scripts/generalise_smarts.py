"""
generalise_smarts.py
====================
Systematically replace bare element symbols in the environment (recursive $(...))
part of SMARTS strings in the Naef & Acree parameter CSV files.

Rules
-----
- C  → [#6]   N  → [#7]   O  → [#8]   S  → [#16]   P  → [#15]
- After substitution: =[#6] → =,:[#6]  and  =[#7] → =,:[#7]
  (so delocalized amidine / enamine C=N / C=C bonds match in RDKit)
  =[#8] and =[#16] are LEFT as-is (carbonyl / thione are always explicit =).
- Atoms already inside [...] (e.g. [CX4], [O-], [#6]) are never touched.
- Two-char atoms Cl, Br, Si, Se are preserved.
- Aromatic atoms c, n, o, s are preserved (lowercase → not in replace map).
- "C aromatic" and "N aromatic" Atom Type entries are EXCLUDED entirely:
  in those entries the bare C/N/O/S neighbour is intentionally aliphatic to
  distinguish it from its aromatic counterpart (c/n/o/s), and changing it
  would cause overlap with the adjacent aromatic entries.
"""

from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# ── Atom replacement map ──────────────────────────────────────────────────────
ATOM_REPLACE = {"C": "[#6]", "N": "[#7]", "O": "[#8]", "S": "[#16]", "P": "[#15]"}
TWO_CHAR = {"Cl", "Br", "Si", "Se", "Te", "Ge"}
SKIP_ATOM_TYPES = {"C aromatic", "N aromatic"}


# ── Core SMARTS transformer ───────────────────────────────────────────────────

def _replace_bare_atoms(fragment: str) -> str:
    """Replace bare element symbols in a SMARTS fragment with [#N] notation.

    Skips atoms already inside [...], two-char atoms (Cl, Br, Si …), and
    lowercase aromatic atoms (c, n, o, s …).
    """
    result: list[str] = []
    i = 0
    n = len(fragment)

    while i < n:
        c = fragment[i]

        # Bracket atom — pass through verbatim
        if c == "[":
            j = i + 1
            depth = 1
            while j < n:
                if fragment[j] == "[":
                    depth += 1
                elif fragment[j] == "]":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            result.append(fragment[i:j])
            i = j
            continue

        # Two-char atoms
        if i + 1 < n and fragment[i : i + 2] in TWO_CHAR:
            result.append(fragment[i : i + 2])
            i += 2
            continue

        # Bare atoms to replace
        if c in ATOM_REPLACE:
            result.append(ATOM_REPLACE[c])
            i += 1
            continue

        # Everything else (bonds =:#~-/, parentheses, ring closures, aromatic atoms, F, I …)
        result.append(c)
        i += 1

    return "".join(result)


def _apply_bond_flexibility(fragment: str) -> str:
    """Change =[#6] → =,:[#6] and =[#7] → =,:[#7].

    Must NOT change =[#8] (carbonyl) or =[#16] (thione) — those bonds are
    always explicit double bonds and should not be made flexible.
    The existing =,:[#6] patterns (with comma already present) are not touched
    because the regex looks for =[#6] immediately (no comma between = and [).
    """
    fragment = re.sub(r"=\[#6\]", "=,:[#6]", fragment)
    fragment = re.sub(r"=\[#7\]", "=,:[#7]", fragment)
    return fragment


def transform_smarts(smarts: str) -> str:
    """Transform one SMARTS string (expected form: [MAIN_ATOM$(RECURSIVE)]).

    Returns the transformed string, or the original if it cannot be parsed.
    """
    smarts = smarts.strip()
    if not smarts or smarts[0] != "[":
        return smarts

    # Find matching ] for the outermost [
    depth = 0
    outer_end = -1
    for idx, ch in enumerate(smarts):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                outer_end = idx
                break
    if outer_end == -1:
        return smarts  # malformed

    outer_content = smarts[1:outer_end]
    trailing = smarts[outer_end + 1 :]  # usually empty

    # Find $( inside the outer content
    dollar_idx = outer_content.find("$(")
    if dollar_idx == -1:
        return smarts  # no recursive SMARTS

    main_spec = outer_content[:dollar_idx]  # e.g. "CH3", "NH1", "N+"

    # Find the matching ) for $(
    rec_start = dollar_idx + 2
    depth = 1
    j = rec_start
    inner = outer_content
    while j < len(inner) and depth > 0:
        if inner[j] == "(":
            depth += 1
        elif inner[j] == ")":
            depth -= 1
        j += 1
    rec_end = j - 1  # position of closing )
    recursive = inner[rec_start:rec_end]
    after_rec = inner[rec_end + 1 :]  # after ) but still inside outer []

    # Transform
    transformed = _replace_bare_atoms(recursive)
    transformed = _apply_bond_flexibility(transformed)

    return "[" + main_spec + "$(" + transformed + ")" + after_rec + "]" + trailing


def validate_smarts(smarts: str) -> bool:
    """Return True if RDKit can parse this SMARTS."""
    try:
        mol = Chem.MolFromSmarts(smarts)
        return mol is not None
    except Exception:
        return False


# ── CSV processing ────────────────────────────────────────────────────────────

def process_csv(csv_path: Path, dry_run: bool = False) -> list[tuple]:
    """Apply SMARTS transformations to a CSV file in-place.

    Returns list of (entry, neighbours, old_smarts, new_smarts) for each change.
    """
    rows: list[dict] = []
    fieldnames: list[str] = []
    changes: list[tuple] = []
    invalid: list[tuple] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))

    for row in rows:
        atom_type = (row.get("Atom Type") or "").strip()
        if atom_type in SKIP_ATOM_TYPES:
            continue  # intentional bare C/N/O/S — skip

        smarts = (row.get("SMARTS") or "").strip()
        if not smarts or smarts in ("None",):
            continue

        new_smarts = transform_smarts(smarts)
        if new_smarts == smarts:
            continue  # no change

        # Validate the new SMARTS before accepting
        if not validate_smarts(new_smarts):
            invalid.append((row.get("Entry", "?"), row.get("Neighbours", ""), smarts, new_smarts))
            continue  # keep original

        changes.append((row.get("Entry", "?"), row.get("Neighbours", ""), smarts, new_smarts))
        row["SMARTS"] = new_smarts

    if invalid:
        print(f"  WARNING: {len(invalid)} SMARTS failed validation and were NOT changed:")
        for entry, nb, old, new in invalid:
            print(f"    E{entry} {nb}: {old!r} → {new!r}")

    if not dry_run:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return changes


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "kawow" / "data"

    targets = [
        data_dir / "naef2024_logkow_parameters.csv",
        data_dir / "naef2024_logkoa_parameters.csv",
        data_dir / "naef2024_logkoa_parameters_recalibrated.csv",
    ]

    for csv_path in targets:
        if not csv_path.exists():
            print(f"SKIP (not found): {csv_path.name}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {csv_path.name}")
        print(f"{'='*60}")

        changes = process_csv(csv_path)

        if not changes:
            print("  No changes needed.")
            continue

        print(f"  {len(changes)} entries changed:")
        print(f"  {'Entry':>5}  {'Neighbours':<22}  {'Old SMARTS':<45}  New SMARTS")
        print(f"  {'-'*5}  {'-'*22}  {'-'*45}  {'-'*45}")
        for entry, nb, old, new in changes:
            print(f"  E{entry:>4}  {nb:<22}  {old:<45}  {new}")

    print("\nDone.")


if __name__ == "__main__":
    main()
