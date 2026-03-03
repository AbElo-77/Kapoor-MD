import os, json 
from pathlib import Path

import numpy as np
from scipy.special import rel_entr

def kl_divergence_2d_hist(samples_p, samples_q, bins=10):

    range_combined = [
        [min(np.min(samples_p[:, 0]), np.min(samples_q[:, 0])), 
         max(np.max(samples_p[:, 0]), np.max(samples_q[:, 0]))],
        [min(np.min(samples_p[:, 1]), np.min(samples_q[:, 1])), 
         max(np.max(samples_p[:, 1]), np.max(samples_q[:, 1]))]
    ]

    hist_p, _ = np.histogramdd(samples_p, bins=bins, range=range_combined, density=True)
    hist_q, _ = np.histogramdd(samples_q, bins=bins, range=range_combined, density=True)

    p = hist_p.flatten()
    q = hist_q.flatten()

    p = p / p.sum()
    q = q / q.sum()
    
    q = np.maximum(q, 1e-10)
    kl_div_elements = rel_entr(p, q)

    return np.sum(kl_div_elements)

def main(mpnn_dir, cons_dir): 
    divergence = {}
    for _, _, files in os.walk(cons_dir):
        for file in files: 

            with open(mpnn_dir / file, 'r') as jf:
                mpnn_dist = json.load(jf)

            with open(cons_dir / file, 'r') as jf: 
                cons_dist = json.load(jf)

            div = kl_divergence_2d_hist(mpnn_dist, cons_dist)
            divergence[Path(file).stem] = div
    
    with open(os.curdir / 'divergence.json', 'w') as jf: 
        json.dump(divergence, jf, indent=4)

if __name__ == "__main__":
    
    MPNN_DIR = Path("lanm_pipeline/mpnn/outputs/seqs/heatmaps/raw")
    CONS_DIR = Path("lanm_pipeline/clusters/inputs/analytics/conservation/raw")

    main(MPNN_DIR, CONS_DIR)