import os, subprocess, json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

def generate_distribution(alignment):
    aln_len = alignment.get_alignment_length()
    n_seqs = len(alignment)

    alphabet = "ACDEFGHIKLMNPQRSTVWY-"
    matrix = [[] for _ in range(len(alphabet))]

    for i in range(aln_len):
        column = str(alignment[:, i]).upper()
        counts = Counter(column)
        for idx, aa in enumerate(alphabet):
            matrix[idx].append(counts.get(aa, 0) / n_seqs)

    return matrix

def generate_conservation(alignment_path: Path, out_dir: Path): 
    for root, _, files in os.walk(alignment_path / 'alignment'): 
        for file in files: 

            alignment = AlignIO.read(Path(root) / file, "fasta")
            n_seqs = len(alignment)
            aln_len = alignment.get_alignment_length()

            consensus_seq = []
            for i in range(aln_len):
                column = alignment[:, i]
                residues = [r for r in column if r != "-"]
                if not residues:
                    consensus_seq.append("-")
                else:
                    most_common_res = Counter(residues).most_common(1)[0][0]
                    consensus_seq.append(most_common_res)

            binary_matrix = np.zeros((n_seqs, aln_len))

            for row_idx, record in enumerate(alignment):
                for col_idx, res in enumerate(record.seq):
                    if res == consensus_seq[col_idx] and res != "-":
                        binary_matrix[row_idx, col_idx] = 1
                    else:
                        binary_matrix[row_idx, col_idx] = 0

            # plt.figure(figsize=(18, n_seqs * 0.3))

            # current_cmap = plt.cm.viridis.copy()
            # current_cmap.set_bad(color='white') 

            # im = plt.imshow(map_data, aspect='auto', cmap=current_cmap, interpolation='nearest')

            # plt.title("Conservation Map", fontsize=14)
            # plt.xlabel("Alignment Position", fontsize=12)
            # plt.ylabel("Sequences", fontsize=12)

            # plt.yticks(range(n_seqs), [rec.id for rec in alignment], fontsize=8)

            # cbar = plt.colorbar(im, orientation='vertical', fraction=0.02, pad=0.02)
            # cbar.set_label('Conservation Score')

            plt.figure(figsize=(18, n_seqs * 0.3))
            plt.imshow(binary_matrix, aspect='auto', cmap='Grays', interpolation='nearest')

            plt.title("Consensus Map")
            plt.xlabel("Alignment Position")
            plt.ylabel("Sequences")
            plt.yticks(range(n_seqs), [rec.id for rec in alignment], fontsize=8)
            
            plt.tight_layout()
            plt.savefig(out_dir / f"{(Path(root) / file).stem}.png", dpi=300)

            data = generate_distribution(alignment)
            with open(out_dir / "raw" / f"{(Path(root) / file).stem}.json", 'w') as jf: 
                    json.dump(data, jf, indent=4)
            

def main(fasta_path, out_dir, cons_dir):
    seqs = {}

    for header, seq in read_fasta(fasta_path):
        if os.path.exists(out_dir / "pdbs" / f"{header}"):
            continue

        result_handle = NCBIWWW.qblast("blastp", "pdb", seq)
        blast_record = NCBIXML.read(result_handle)

        pdb_ids = []
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                if hsp.expect < 0.001:
                    pdb_id = alignment.title.split("|")[1] 
                    pdb_ids.append(pdb_id.upper())
                    
        pdbl = PDBList(server='https://files.rcsb.org')
        pdb_ids = list(set(str(pdb_id).upper() for pdb_id in pdb_ids))
        os.makedirs(out_dir / "pdbs" / f"{header}", exist_ok=True)
        
        for pdb_id in pdb_ids:
            pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir= out_dir / "pdbs" / f"{header}")

        fasta = f">{header}\n{seq}"
        file = StringIO(fasta)

        record = list(SeqIO.parse(file, "fasta"))
        recs = [record[0]]

        for root, _, files in os.walk(out_dir / "pdbs" / f"{header}"):
            for file in files: 
                if os.path.getsize(Path(root) / file) == 0:
                    continue

                record = list(SeqIO.parse(Path(root) / file, "pdb-seqres"))
                if not record:
                    continue
                if abs(len(record[0]) - len(seq)) > 25: 
                    continue
                
                recs.append(record[0])
        
        seqs[header] = recs

# UNCOMMENT FOR RERUNS
    # for root, headers, _ in os.walk(out_dir / "pdbs"):
    #     for header in headers: 
    #         recs = []
    #         for root2, _, files in os.walk(Path(root) / header):
    #             for file in files: 

    #                 if os.path.getsize(Path(root2) / file) == 0:
    #                     continue

    #                 record = list(SeqIO.parse(Path(root2) / file, "pdb-seqres"))
    #                 if not record:
    #                     continue
    #                 if abs(len(record[0]) - len(seq)) > 10: 
    #                     continue
                    
    #                 recs.append(record[0])
        
    #         seqs[header] = recs
        
    # for header, recs in seqs.items():
    #     SeqIO.write(recs, out_dir / "pdbs" / f"{header}" / "seqs.fa", "fasta")

    #     with open(out_dir / "alignment" / f"{header}.fa", "w") as out_file:
    #         subprocess.run(["mafft", 
    #                         "--auto", 
    #                         out_dir / "pdbs" / f"{header}" / "seqs.fa"], 
    #                         stdout=out_file, 
    #                         check=True)

    generate_conservation(out_dir, cons_dir)

# -------- CONSERVATION SCORING ------- #

# -------- KL DIVERGENCE CALCULATION -------- #

if __name__ == "__main__":

    FASTA = Path("lanm_pipeline/clusters/inputs/top_seqs.fa")
    OUT_DIR = Path("lanm_pipeline/clusters/inputs/analytics/msa")
    CONS_DIR = Path("lanm_pipeline/clusters/inputs/analytics/conservation")

    main(FASTA, OUT_DIR, CONS_DIR)
