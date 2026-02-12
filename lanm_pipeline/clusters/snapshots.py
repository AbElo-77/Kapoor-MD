import os
from pathlib import Path

from typing import List, Tuple
import argparse

import pymol
from pymol import cmd

def read_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    with fasta_path.open("r") as f:
        header = ""
        seq_lines: List[str] = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    entries.append((header, "".join(seq_lines).upper()))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:
            entries.append((header, "".join(seq_lines).upper()))
    return entries

def render_protein(pdb_path, ref_path, output_path):
    ref_name = "reference_obj" 
    obj_name = "current_prot"

    if ref_name not in cmd.get_object_list():
        cmd.load(str(ref_path), ref_name)

    cmd.delete("all")

    cmd.load(str(ref_path), ref_name)
    cmd.load(str(pdb_path), obj_name)

    if cmd.count_atoms(obj_name) == 0:
        print(f"Error: No atoms found in {pdb_path}")
        return

    cmd.super(obj_name, ref_name)

    cmd.show_as("cartoon", obj_name)
    cmd.hide("everything", ref_name)

    cmd.set_color("plddt_very_high", [0.00, 0.32, 0.84])
    cmd.set_color("plddt_high",      [0.40, 0.79, 0.94])
    cmd.set_color("plddt_confident", [1.00, 0.86, 0.25])
    cmd.set_color("plddt_low",       [1.00, 0.49, 0.27])
    try:
        cmd.spectrum("b", "plddt_low plddt_confident plddt_high plddt_very_high", 
                     selection=obj_name, minimum=50, maximum=90)
    except:
        cmd.color("gray70", obj_name)

    cmd.set("ray_opaque_background", 0)
    cmd.orient(ref_name)
    cmd.zoom(ref_name, buffer=2.0)

    cmd.ray(1200, 1200)
    cmd.png(str(output_path), dpi=300)


def main():

    FASTA: Path = Path("lanm_pipeline/clusters/inputs/top_seqs.fa")
    PDB_PATH: Path = Path("lanm_pipeline/af_esm/outputs/structure_preds")
    OUT_DIR: Path = Path("lanm_pipeline/clusters/snapshots/")

    REF_PATH: Path = Path("lanm_pipeline/mpnn/src/pdb/3CLN.pdb")

    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--fasta", default=FASTA, type=Path, help="FASTAs of sequences to cluster.")
    argparser.add_argument("--pdb_path", default=PDB_PATH, type=Path, help="PDBs of sequences to cluster.")
    argparser.add_argument("--ref_path", default=REF_PATH, type=Path, help="Reference PDB for performing alignment.")
    argparser.add_argument("--out_dir", default=OUT_DIR, type=Path, help="Output PNGs.")
    args = argparser.parse_args()

    FASTA = args.fasta
    PDB_PATH = args.pdb_path
    OUT_DIR = args.out_dir
    
    REF_PATH = args.ref_path

    pymol.finish_launching(['pymol', '-cq']) 
    fastas = read_fasta(FASTA)

    cmd.load(str(REF_PATH), "reference_obj")
    cmd.hide("everything", "reference_obj")

    for fasta in fastas:
        pdb = PDB_PATH / fasta[0] / "predicted.pdb"
        render_protein(pdb, REF_PATH, OUT_DIR / (fasta[0] + ".png"))

    cmd.quit()

if __name__ == '__main__':
    main()
