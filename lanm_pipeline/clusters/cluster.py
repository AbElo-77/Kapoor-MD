from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

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

def compute_result(header: str, seq: str) -> Result:
    q, lam = seq_to_charge_lambda(seq)
    scd_val = scd_fft(q)
    shd_val = shd(lam)
    nu_val = eq_nu(shd_val, scd_val)

    net_q = float(q.sum())
    abs_q = float(np.abs(q).sum())
    f_pos = float(q[q > 0].sum())
    f_neg = float(q[q < 0].sum()) 
    mean_lambda = float(lam.mean())

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
    )

# --------------------------------------------

def cluster_rows(rows: List[Result], k: int, seed: int = 0) -> np.ndarray:

    X = np.array([
        [r.scd, r.shd, r.nu, r.mean_lambda, r.net_q, r.abs_q, r.length]
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

def write_table(out_csv: Path, rows: List[Result], labels: np.ndarray) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("header,cluster,length,scd,shd,nu,mean_lambda,net_q,abs_q,f_pos,f_neg\n")
        for r, c in zip(rows, labels):
            f.write(
                f"{r.header},{int(c)},{r.length},"
                f"{r.scd:.6f},{r.shd:.6f},{r.nu:.6f},{r.mean_lambda:.6f},"
                f"{r.net_q:.3f},{r.abs_q:.3f},{r.f_pos:.3f},{r.f_neg:.3f}\n"
            )

# --------------------------------------------

def main():

    FASTA: Path = Path("lanm_pipeline/clusters/inputs/top_seqs.fa")
    OUT_CSV: Path = Path("lanm_pipeline/clusters/scd_clusters/clustered.csv")
    K = 2

    argparser = argparse.ArgumentParser(description="Cluster sequences using SCD/SHD descriptors.")
    argparser.add_argument("--fasta", default=FASTA, type=Path, help="FASTAs of sequences to cluster.")
    argparser.add_argument("--out_csv", default=OUT_CSV, type=Path, help="Output CSV with descriptors + cluster labels.")
    argparser.add_argument("--k", type=int, default=K, help="Number of clusters for MiniBatchKMeans.")
    args = argparser.parse_args()

    FASTA = args.fasta
    OUT_CSV = args.out_csv
    K = args.k

    rows: List[Result] = []
    for header, seq in read_fasta(FASTA):
        rows.append(compute_result(header, seq))

    labels = cluster_rows(rows, k=K, seed=0)
    write_table(OUT_CSV, rows, labels)


if __name__ == "__main__":
    main()
