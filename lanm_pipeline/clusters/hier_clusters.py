import os
from pathlib import Path
from typing import List, Tuple

from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator

from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform

import numpy as np
import argparse

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

def cluster_hier(fasta_path: Path, k: int, seed: int) -> np.ndarray: 
    aln = AlignIO.read(fasta_path, 'fasta')

    calculator = DistanceCalculator('blosum62')
    dist_matrix = calculator.get_distance(aln)

    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method='complete')
    labels = cut_tree(Z, n_clusters=k).flatten()

    return labels

def make_dirs(out_dir: Path, fasta_path: Path, labels: np.ndarray): 
    dist_lab = set(labels.tolist())
    fastas = read_fasta(fasta_path)

    for val in dist_lab: 
        (out_dir / f'hier_{val}').mkdir(parents=True, exist_ok=True)
        with open(out_dir / f'hier_{val}' / 'seqs.fa', 'w') as f:
            for idx, label in enumerate(labels.tolist()):
                if label == val:
                    f.write(f">{fastas[idx][0]}\n{fastas[idx][1]}\n")
                
            f.close()

def main():

    FASTA: Path = Path("lanm_pipeline/clusters/inputs/top_seqs.fa")
    PDB_PATH: Path = Path("lanm_pipeline/af_esm/outputs/structure_preds")
    OUT_DIR: Path = Path("lanm_pipeline/clusters/clusters")
    K = 5

    argparser = argparse.ArgumentParser(description="Cluster sequences using SCD/SHD descriptors.")
    argparser.add_argument("--fasta", default=FASTA, type=Path, help="FASTAs of sequences to cluster.")
    argparser.add_argument("--pdb_path", default=PDB_PATH, type=Path, help="PDB output folder.")
    argparser.add_argument("--out_dir", default=OUT_DIR, type=Path, help="Output CSV with descriptors + cluster labels.")
    argparser.add_argument("--k", type=int, default=K, help="Number of clusters for MiniBatchKMeans.")
    args = argparser.parse_args()

    FASTA = args.fasta
    out_dir = args.out_dir
    K = args.k

    labels = cluster_hier(FASTA, k=K, seed=0)
    make_dirs(OUT_DIR, FASTA, labels)

if __name__ == "__main__": 
    main()