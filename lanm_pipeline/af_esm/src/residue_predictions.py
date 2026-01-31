import numpy as np

import argparse
import os, json
import pickle

import shutil
import subprocess
from pathlib import Path
import tempfile

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

ESMBIND_ROOT = Path("lanm_pipeline/af_esm/src/ESMBind")
PY = "python3"

AA_WEIGHTS = {
    "ASP": 1.00, "GLU": 1.00, "ASN": 0.50, "GLN": 0.50, "SER": 0.35, 
    "THR": 0.35,"TYR": 0.30, "HIS": 0.10, "CYS": 0.10, "ALA": 0.05, 
    "GLY": 0.05,"PRO": 0.05, "ARG": 0.02, "LYS": 0.02, "ILE": 0.01, 
    "LEU": 0.01, "VAL": 0.01, "MET": 0.01, "PHE": 0.01,"TRP": 0.01,
}


seq1to3 = {
    "R":"ARG","H":"HIS","K":"LYS","D":"ASP","E":"GLU",
    "S":"SER","T":"THR","N":"ASN","Q":"GLN","C":"CYS",
    "G":"GLY","P":"PRO","A":"ALA","V":"VAL",
    "I":"ILE","L":"LEU","M":"MET","F":"PHE","Y":"TYR","W":"TRP",
}

def run(cmd, cwd=None):
    print(">>", " ".join(map(str, cmd)))
    try:
        subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)

def scores_to_binary(scores: np.ndarray, frac: float = 0.05, min_k: int = 5):
    L = scores.shape[0]
    k = max(min_k, int(np.ceil(frac * L)))

    thresh = np.partition(scores, -k)[-k]
    mask = (scores >= thresh).astype(np.uint8)

    if mask.sum() > k:
        idx = np.argsort(scores)[::-1]
        mask[:] = 0
        mask[idx[:k]] = 1

    return mask

def get_residues(fasta_path: Path, pdb_path: Path) -> str:
    """
    Need to precompute binary residue labels based dually on AA type and RSA from DSSP.
    """
    with open(fasta_path, "r") as f:
        lines = f.readlines()

    seq = lines[1].strip().upper()
    L = len(seq)

    scores = np.full(L, 0.05, dtype=np.float32)
    for i, aa in enumerate(seq):
        scores[i] = AA_WEIGHTS.get(seq1to3[aa], 0.05) 

    p = PDBParser(QUIET=True)
    structure = p.get_structure("protein", str(pdb_path))
    model = structure[0]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb') as tmp:
        with open(pdb_path, 'r') as original:
            tmp.write("HEADER    PROTEIN                                 \n")
            tmp.write(original.read())
        tmp.flush()

        try:
            dssp = DSSP(model, tmp.name, dssp='mkdssp')
        except Exception as e:
            print(f"DSSP Error: {e}")
            return None

    rsa_vals = np.array([dssp[k][3] for k in dssp.keys()], dtype=np.float32)

    n = min(L, rsa_vals.shape[0])
    threshold = 0.20
    rsa_weight = 1.0 / (1.0 + np.exp(-12.0 * (rsa_vals[:n] - threshold)))
    scores[:n] *= rsa_weight

    mask = scores_to_binary(scores, frac=0.05, min_k=3)

    binary_labels = "".join("1" if x else "0" for x in mask.tolist())
    return binary_labels

def parse_binary(binary_path: Path, seq_id: str) -> str:
    with open(binary_path, "r") as f:
        lines = f.readlines()

    binary = lines.index(f">{seq_id}\n") + 1
    if not binary:
        raise ValueError(f"Sequence ID {seq_id} not found in binary file {binary_path}")

    return lines[binary].strip()
    
def build_inputs(fasta_dir: Path, af_out_root: Path, workdir: Path) -> tuple[Path, Path]:

    workdir.mkdir(parents=True, exist_ok=True)
    pdb_dir = workdir / "pdbs"
    pdb_dir.mkdir(exist_ok=True)

    fasta_out = workdir / "sequences.fasta"
    labels_out = workdir / "residue_labels.txt"
    with fasta_out.open("w") as w:
        for fa in sorted(list(fasta_dir.rglob("*.fa"))):
            
            af_pdb = af_out_root / fa.stem / "predicted.pdb"
            if not af_pdb.exists():
                continue
            
            seq_id = fa.stem
            seq = "".join(
                line.strip() for line in fa.read_text().splitlines()
                if line and not line.startswith(">")
            ).upper()
            seq = "".join([c for c in seq if "A" <= c <= "Z"])
            if not seq:
                continue
            w.write(f">{seq_id}\n{seq}\n")

            with open(labels_out, "a") as a:
                a.write(f">{seq_id}\n")
                 
                residue_binary = get_residues(fa, af_pdb)
                a.write(f"{residue_binary}\n")

            shutil.copyfile(af_pdb, pdb_dir / f"{seq_id}.pdb")

    return (fasta_out, pdb_dir)

def main(
    fasta_dir: Path,
    af_out_root: Path,
    workdir: Path,
    ion: str = "CA",
):
    
    fasta_path, pdb_dir = build_inputs(fasta_dir, af_out_root, workdir)
    output_root = Path("lanm_pipeline/af_esm/outputs/esm_embeds")

    run([PY, ESMBIND_ROOT / "multi_modal_binding/get_esm_embedding.py", fasta_path, output_root / "esm"], cwd=os.curdir)
    run([PY, ESMBIND_ROOT / "multi_modal_binding/get_esm_if_embedding.py", fasta_path, output_root / "esm_if", pdb_dir], cwd=os.curdir)

    config_path = ESMBIND_ROOT / "multi_modal_binding/configs/inference.json"
    run([PY, ESMBIND_ROOT / "multi_modal_binding/inference.py", "--config", config_path], cwd=os.curdir)

    results_dir = Path("lanm_pipeline/af_esm/outputs/residue_preds")
    pkls = sorted(results_dir.rglob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pkls:
        raise FileNotFoundError(f"No .pkl outputs found in {results_dir}")
    pkl_path = pkls[0]

    with pkl_path.open("rb") as f:
        preds = pickle.load(f)

    for seq_id, preds in preds['CA'].items():
        binding_sites = [i for i, p in enumerate(preds) if p > 0.5]
    
        if len(binding_sites) < 10 or float(np.mean([preds[i] for i in binding_sites])) < 0.6:
            continue

        metrics = {
            "id": seq_id,
            "sites": binding_sites, 
            "num_sites": len(binding_sites),
            "avg_score": float(np.mean([preds[i] for i in binding_sites])),
        }

        json_path = results_dir / f"{seq_id}" / "CA_preds.json"
        os.makedirs(json_path.parent, exist_ok=True)

        with json_path.open("w") as jf:
            json.dump(metrics, jf, indent=4)

    return preds, pkl_path

if __name__ == "__main__":

    FASTA_DIR = "lanm_pipeline/mpnn/outputs/fastas/"
    AF_ROOT  = "lanm_pipeline/af_esm/outputs/structure_preds/"
    WORK_DIR  = "lanm_pipeline/af_esm/esm_inputs/"
    ION = "CA"

    build_inputs(Path(FASTA_DIR), Path(AF_ROOT), Path(WORK_DIR))

    argparser = argparse.ArgumentParser(description="Run ESMBind calcium binding site predictions")
    argparser.add_argument("--fasta_dir", default=FASTA_DIR, help="Path to input FASTA directory")
    argparser.add_argument("--af_root", default=AF_ROOT, help="Path to AlphaFold output root directory")
    argparser.add_argument("--workdir", default=WORK_DIR, help="Path to working directory for ESMBind inputs/outputs")
    argparser.add_argument("--ion", default=ION, help="Ion type for binding site prediction")

    args = argparser.parse_args()
    FASTA_DIR = Path(args.fasta_dir)
    AF_ROOT = Path(args.af_root)
    WORK_DIR = Path(args.workdir)
    ION = args.ion

    preds, pkl_path = main(FASTA_DIR, AF_ROOT, WORK_DIR, ion=ION)
    print("Loaded predictions from:", pkl_path)