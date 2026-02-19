from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio import SeqIO

import numpy as np
import csv 

import pandas as pd
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

aa_param = """#AA     Mass    Charge  Sigma   Lambda
ALA     71.08   0.00    5.040   0.603
ARG     156.20  1.00    6.560   0.559
ASN     114.10  0.00    5.680   0.588
ASP     115.10  -1.00   5.580   0.294
CYS     103.10  0.00    5.480   0.647
GLN     128.10  0.00    6.020   0.559
GLU     129.10  -1.00   5.920   0.000
GLY     57.05   0.00    4.500   0.573
HIS     137.10  0.00    6.080   0.765
ILE     113.20  0.00    6.180   0.706
LEU     113.20  0.00    6.180   0.720
LYS     128.20  1.00    6.360   0.382
MET     131.20  0.00    6.180   0.676
PHE     147.20  0.00    6.360   0.823
PRO     97.12   0.00    5.560   0.759
SER     87.08   0.00    5.180   0.588
THR     101.10  0.00    5.620   0.588
TRP     186.20  0.00    6.780   1.000
TYR     163.20  0.00    6.460   0.897
VAL     99.07   0.00    5.860   0.665
"""

seq1to3 = {
    "R":"ARG","H":"HIS","K":"LYS","D":"ASP","E":"GLU",
    "S":"SER","T":"THR","N":"ASN","Q":"GLN","C":"CYS",
    "G":"GLY","P":"PRO","A":"ALA","V":"VAL",
    "I":"ILE","L":"LEU","M":"MET","F":"PHE","Y":"TYR","W":"TRP",
}

AA: Dict[str, np.ndarray] = {}
for line in aa_param.splitlines():
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    AA[parts[0]] = np.array(list(map(float, parts[1:])), dtype=np.float64)

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

# --------------------------------------------

def _dssp(header: str, ref_path) -> int: 

    p = PDBParser()
    structure = p.get_structure("ID", ref_path)
    model = structure[0]
    dssp = DSSP(model, ref_path, dssp="mkdssp", file_type="PDB")

    total = len(dssp)

    ss_codes = [res[2] for res in dssp]
    helix = ss_codes.count('H') / total
    sheet = ss_codes.count('E') / total
    loop = (ss_codes.count('-') + ss_codes.count('C')) / total
    
    features = [helix, sheet, loop]
    return features

def seq_to_charge_lambda(seq: str) -> Tuple[np.ndarray, np.ndarray]:
    n = len(seq)
    q = np.zeros(n, dtype=np.float64)
    lam = np.zeros(n, dtype=np.float64)
    for i, aa1 in enumerate(seq):
        aa3 = seq1to3.get(aa1)
        if aa3 is None or aa3 not in AA:
            raise ValueError(f"unknown residue: {aa1}")
        q[i] = AA[aa3][1]
        lam[i] = AA[aa3][3]
    return q, lam

def scd_fft(q: np.ndarray) -> float:
    
    n = q.size
    if n < 2:
        return 0.0
    
    m = 1 << (2*n - 1).bit_length()
    fq = np.fft.rfft(q, m)
    ac = np.fft.irfft(fq * np.conj(fq), m).real[:n]  
    d = np.arange(1, n, dtype=np.float64)
    return float(np.sum(np.sqrt(d) * ac[1:]) / n)

def shd(lam: np.ndarray) -> float:
    n = lam.size
    if n < 2:
        return 0.0

    k = np.arange(1, n, dtype=np.float64)
    H = np.zeros(n, dtype=np.float64)
    H[1:] = np.cumsum(1.0 / k)


    i = np.arange(n, dtype=np.int64)
    weights = H[i] + H[n - 1 - i]
    return float(np.sum(lam * weights) / n)

def eq_nu(shd_val: float, scd_val: float) -> float:
    return float(-0.0423 * shd_val + 0.00740 * scd_val + 0.701)

@dataclass
class Result:
    header: str
    length: int
    scd: float
    shd: float
    nu: float
    mean_lambda: float
    net_q: float
    abs_q: float
    f_pos: float
    f_neg: float
    helix_percent: float = None
    sheet_percent: float = None
    loop_percent: float = None

def compute_result(header: str, seq: str, ref_path: Path, dssp=True) -> Result:
    q, lam = seq_to_charge_lambda(seq)
    scd_val = scd_fft(q)
    shd_val = shd(lam)
    nu_val = eq_nu(shd_val, scd_val)

    net_q = float(q.sum())
    abs_q = float(np.abs(q).sum())
    f_pos = float(q[q > 0].sum())
    f_neg = float(q[q < 0].sum()) 
    mean_lambda = float(lam.mean())

    if dssp:
        dssp = _dssp(header, ref_path)

        return Result(
            header=header,
            length=len(seq),
            scd=scd_val,
            shd=shd_val,
            nu=nu_val,
            mean_lambda=mean_lambda,
            net_q=net_q,
            abs_q=abs_q,
            f_pos=f_pos,
            f_neg=f_neg,
            helix_percent=dssp[0], 
            sheet_percent=dssp[1], 
            loop_percent=dssp[2]
        )
    else: 
        return Result(
            header=header,
            length=len(seq),
            scd=scd_val,
            shd=shd_val,
            nu=nu_val,
            mean_lambda=mean_lambda,
            net_q=net_q,
            abs_q=abs_q,
            f_pos=f_pos,
            f_neg=f_neg, 
            helix_percent=None, 
            sheet_percent=None, 
            loop_percent=None
        )

# --------------------------------------------

def cluster_rows(rows: List[Result], k: int, seed: int = 0, val: string = "scd") -> np.ndarray:

    if val == "scd":
        X = np.array([
            [r.scd]
            for r in rows
        ], dtype=np.float64)
    if val == "q_net":
        X = np.array([
            [r.net_q]
            for r in rows
        ], dtype=np.float64)
    if val == "dssp":
        X = np.array([
            [r.helix_percent, r.sheet_percent, r.loop_percent]
            for r in rows
        ], dtype=np.float64)

    Xs = StandardScaler().fit_transform(X)

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=4096,
        n_init="auto",
        max_no_improvement=50,
        reassignment_ratio=0.01,
        verbose=0,
    )
    labels = km.fit_predict(Xs)
    return labels

def write_row_vs_loops(output_path: Path, combined_df: pd.DataFrame, rows_only_loops: List[Result]):
    loop_lookup = {r.header: r for r in rows_only_loops}
    
    fieldnames = [
        'header', 'clusters_(scd-q_net-hier)',
        'full_scd', 'loop_scd', 
        'full_net_q', 'loop_net_q'
    ]
    
    with open(output_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in combined_df.iterrows():
            header_id = row['header']
            loop_res = loop_lookup.get(header_id)
            
            if loop_res:
                writer.writerow({
                    'header': header_id,
                    'clusters_(scd-q_net-hier)': row['combined_cluster'],
                    'full_scd': round(row['scd'], 4),
                    'loop_scd': round(loop_res.scd, 4),
                    'full_net_q': row['net_q'],
                    'loop_net_q': loop_res.net_q,
                })

def choose_representatives(rows_v_loops: pd.DataFrame, ref_metrics: Result, output_csv: Path):
    features = ['full_scd', 'loop_scd', 'full_net_q']
    scaler = StandardScaler()
    X = scaler.fit_transform(rows_v_loops[features])

    selections = []

    def add_selection(idx, reason):
        res = rows_v_loops.iloc[idx].copy()
        res['selection_reason'] = reason
        selections.append(res)

    wt_vector = scaler.transform([[ref_metrics["scd"], 0, ref_metrics["net_q"]]])
    closest_idx, _ = pairwise_distances_argmin_min(wt_vector, X)
    add_selection(closest_idx[0], "Best Mimic")

    add_selection(rows_v_loops['loop_scd'].idxmax(), "Max Loop SCD")

    rows_v_loops['scaffold_diff'] = rows_v_loops['full_scd'] - rows_v_loops['loop_scd']
    add_selection(rows_v_loops['scaffold_diff'].idxmax(), "Max Scaffold Contribution")

    rows_v_loops['q_dist'] = (rows_v_loops['full_net_q'] - ref_metrics["net_q"]).abs()
    add_selection(rows_v_loops['q_dist'].idxmin(), "Closest Net Q")

    centroid = X.mean(axis=0).reshape(1, -1)
    medoid_idx, _ = pairwise_distances_argmin_min(centroid, X)
    add_selection(medoid_idx[0], "Average Mimic")

    final_df = pd.DataFrame(selections).drop_duplicates(subset=['header'])
    final_df.to_csv(output_csv, index=False)
    
    print(f"Successfully saved {len(final_df)} unique representatives to {output_csv}")

# --------------------------------------------

def main():

    FASTA: Path = Path("lanm_pipeline/clusters/inputs/top_seqs.fa")
    CLUSTERS: Path = Path("lanm_pipeline/clusters/clusters")
    REF_PATH: Path = Path("lanm_pipeline/mpnn/src/pdb/3CLN.pdb")
    OUT_DIR: Path = Path("lanm_pipeline/clusters/clusters/representatives")

    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--fasta", default=FASTA, type=Path, help="FASTAs of sequences to cluster.")
    argparser.add_argument("--clusters", default=CLUSTERS, type=Path, help="Directory containing cluster CSVs.")
    argparser.add_argument("--ref_path", default=REF_PATH, type=Path, help="Reference PDB for performing alignment.")
    argparser.add_argument("--out_dir", default=OUT_DIR, type=Path, help="Directory to output representative selections.")
    args = argparser.parse_args()

    FASTA = args.fasta
    CLUSTERS = args.clusters
    REF_PATH = args.ref_path
    OUT_DIR = args.out_dir

    records = list(SeqIO.parse(REF_PATH, "pdb-seqres"))
    row_ref = compute_result("3CLN", records[0].seq, REF_PATH)

    binding_loops = [
        range(20, 32), range(56, 68), 
        range(93, 105), range(129, 141)
    ]
    binding_indices = {i for r in binding_loops for i in r}

    df_scd = pd.read_csv(CLUSTERS / "scd.csv")
    df_qnet = pd.read_csv(CLUSTERS / "q_net.csv")
    df_hier = pd.read_csv(CLUSTERS / "hier.csv")

    combined_df = pd.merge(
        df_scd, 
        df_qnet[['header', 'cluster']], 
        on='header', 
        suffixes=('_scd', '_qnet')
    )
  
    combined_df['combined_cluster'] = list(zip(combined_df.cluster_scd, combined_df.cluster_qnet, df_hier.cluster))

    # --------------------------------------------

    modified_list = []
    for idx, aa in enumerate(records[0].seq):
        pos = idx + 1 
        
        if pos in binding_indices:
            modified_list.append(aa)
        else:
            modified_list.append('G')
            
    seq_only_loops = "".join(modified_list)
    row_ref_loops = compute_result("3CLN_loops", seq_only_loops, REF_PATH, dssp=False)

    rows_only_loops: List[Result] = []
    for header, seq in read_fasta(FASTA):

        modified_list = []
        for idx, aa in enumerate(seq):
            pos = idx + 1 
            
            if pos in binding_indices:
                modified_list.append(aa)
            else:
                modified_list.append('G')
                
        seq_only_loops = "".join(modified_list)
        rows_only_loops.append(compute_result(header, seq_only_loops, REF_PATH, dssp=False))
    
    ref_metrics = {
        "scd": row_ref.scd, 
        "scd_loop": row_ref_loops.scd, 
        "net_q": row_ref.net_q, 
        "net_q_loop": row_ref_loops.net_q
    }

    with open(OUT_DIR / "reference.json", "w") as jf: 
        json.dump(ref_metrics, jf, indent=4)

    write_row_vs_loops(OUT_DIR / "row_loops.csv", combined_df, rows_only_loops)
    rows_v_loops = pd.read_csv(OUT_DIR / "row_loops.csv")

    choose_representatives(rows_v_loops, ref_metrics, OUT_DIR / "representatives.csv")

    # --------------------------------------------
    

if __name__ == "__main__":
    main()
