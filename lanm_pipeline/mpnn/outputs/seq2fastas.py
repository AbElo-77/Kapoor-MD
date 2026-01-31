import os
import hashlib

import argparse

SEQ_FOLDER = "lanm_pipeline/mpnn/outputs/seqs"
OUT_DIR = "lanm_pipeline/mpnn/outputs/fastas"

os.makedirs(OUT_DIR, exist_ok=True)

def is_header(line: str) -> bool:
    return line.startswith(">")

def normalize_seq(seq: str) -> str:
    return seq.strip().upper()

def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()[:12]

def generate_outputs(seqs_folder, out_dir):

    global_idx = 0
    for root, _, files in os.walk(seqs_folder):
        for file in files:
            path = os.path.join(root, file)

            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]

            i = 0
            while i < len(lines):
                if not is_header(lines[i]):
                    i += 1
                    continue

                seq_line = lines[i + 1]
                sequences = [normalize_seq(s) for s in seq_line.split("/")]

                for seq in sequences:
                    h = seq_hash(seq)
                    fasta_path = os.path.join(
                        OUT_DIR, f"{h}.fa"
                    )

                    with open(fasta_path, "w") as out:
                        out.write(f">{h}\n{seq}\n")

                    global_idx += 1

                i += 2

if __name__ == "__main__": 

    SEQ_FOLDER = "lanm_pipeline/mpnn/outputs/seqs"
    OUT_DIR = "lanm_pipeline/mpnn/outputs/fastas"

    argparser = argparse.ArgumentParser(description="Generate FASTA files from sequence outputs")
    argparser.add_argument("--seqs_folder", default=SEQ_FOLDER, help="Path to folder containing sequence files")
    argparser.add_argument("--out_dir", default=OUT_DIR, help="Path to output FASTA files")

    args = argparser.parse_args()
    SEQ_FOLDER = args.seqs_folder
    OUT_DIR = args.out_dir

    generate_outputs(SEQ_FOLDER, OUT_DIR)

