import os, subprocess, json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd

from Bio.Blast import NCBIWWW, NCBIXML
from Bio.PDB import PDBList

from io import StringIO
from Bio import SeqIO, AlignIO

from typing import List, Tuple
from collections import Counter
import math

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

def generate_conservation(alignment_path: Path, out_dir: Path): 

    alignment = AlignIO.read(Path(alignment_path) / "alignment.fa", "fasta")
    n_seqs = len(alignment)
    aln_len = alignment.get_alignment_length()

    cmd = [
        "rate4site",
        "-s", Path(alignment_path) / "alignment.fa",
        "-o", out_dir / f"conservation.res",
        '-mj',
        "-ib" 
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )    
    except subprocess.CalledProcessError as e:
        print(f"Error running Rate4Site:\n{e.stderr}")
        return None

    df = pd.read_csv(out_dir / f"conservation.res", sep="\s+", comment='#', header=None, names=["Pos", "Seq", "Score", "QQ", "N/A1", "N/A2"])
    labels = [9, 8, 7, 6, 5, 4, 3, 2, 1]

    jsn = df.to_json()

    df['ConSurf_Grade'] = pd.cut(df['Score'], bins=9, labels=labels)

    discrete_viridis = mpl.colormaps['viridis_r'].resampled(9)
    data_2d = df['Score'].values.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(15, 2))
    im = ax.imshow(data_2d, aspect='auto', cmap=discrete_viridis)

    ax.set_xlabel("Position in Protein Sequence")
    ax.set_title("Confidence Per Position")

    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.3, label="Max Probability")
    plt.savefig(out_dir / "conservation.png", dpi=300)

    with open(out_dir / "conservation.json", 'w') as jf: 
        json.dump(jsn, jf, indent=4)
            

def main(fasta_path, ref_path, out_dir, cons_dir):
    records = list(SeqIO.parse(ref_path, "pdb-seqres"))
    seq, header = records[0].seq, "3CLN"

    result_handle = NCBIWWW.qblast(
        "blastp", 
        "swissprot", 
        seq, 
        hitlist_size=500,
        expect=0.0001
    )

    blast_record = NCBIXML.read(result_handle)
    unique_species = set()

    homolog_sequences = []
    homolog_sequences.append(f">{header}\n{seq}")

    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            
            identity = (hsp.identities / hsp.align_length) * 100
            if identity > 95 or identity < 35:
                continue
                
            try:
                raw_def = alignment.title
                if "[" in raw_def and "]" in raw_def:
                    species = raw_def.split("[")[1].split("]")[0]
                else:
                    species = alignment.hit_id.split("_")[-1]
            except:
                species = "Unknown"

            if species in unique_species:
                continue

            unique_species.add(species)

            seq_data = hsp.sbjct.replace("-", "")
            header = f"{alignment.hit_id} | Identity:{identity:.1f}"
            
            homolog_sequences.append(f">{header}\n{seq_data}")

            if len(homolog_sequences) > 150:
                break

    with open(out_dir / "seqs.fa", "w") as f:
        f.write("\n".join(homolog_sequences))
        
    with open(out_dir / "alignment.fa", "w") as out_file:
        subprocess.run(["mafft", 
                        "--auto", 
                        out_dir / "seqs.fa"], 
                        stdout=out_file, 
                        check=True)

    generate_conservation(out_dir, cons_dir)

if __name__ == "__main__":

    FASTA = Path("lanm_pipeline/clusters/inputs/top_seqs.fa")
    REF_PATH: Path = Path("lanm_pipeline/mpnn/src/pdb/3CLN.pdb") 
    OUT_DIR = Path("lanm_pipeline/clusters/inputs/analytics/msa")
    CONS_DIR = Path("lanm_pipeline/clusters/inputs/analytics/conservation")

    main(FASTA, REF_PATH, OUT_DIR, CONS_DIR)
