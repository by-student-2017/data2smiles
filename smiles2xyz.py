#!/usr/bin/env python3
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_optimized_mol(smiles: str, use_mmff: bool = True, max_iters: int = 200):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol) != 0:
            raise RuntimeError(f"3D embedding failed for: {smiles}")
    if use_mmff and AllChem.MMFFHasAllMoleculeParams(mol):
        res = AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
        if res != 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    return mol

def rdkit_mol_to_xyz(mol, title: str = "molecule"):
    conf = mol.GetConformer()
    natoms = mol.GetNumAtoms()
    lines = [str(natoms), title]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        sym = atom.GetSymbol()
        lines.append(f"{sym} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}")
    return "\n".join(lines) + "\n"

def main():
    input_path = Path("smiles.txt")
    outdir = Path("molecular_files")
    outdir.mkdir(exist_ok=True)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    smiles_list = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            smiles_list.append(s)

    if not smiles_list:
        print("No SMILES found in smiles.txt", file=sys.stderr)
        sys.exit(1)

    for i, smi in enumerate(smiles_list, start=1):
        try:
            mol = smiles_to_optimized_mol(smi, use_mmff=True, max_iters=300)
            title = f"mol_{i}: {smi}"
            # Write XYZ
            xyz_str = rdkit_mol_to_xyz(mol, title=title)
            (outdir / f"mol_{i}.xyz").write_text(xyz_str, encoding="utf-8")
            # Write MOL
            Chem.MolToMolFile(mol, str(outdir / f"mol_{i}.mol"))
            ## Write MOL2
            #Chem.MolToMol2File(mol, str(outdir / f"mol_{i}.mol2"))
            #print(f"Wrote mol_{i}.xyz, mol_{i}.mol, mol_{i}.mol2")
        except Exception as e:
            print(f"[Warning] Skipping entry {i} ({smi}): {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
