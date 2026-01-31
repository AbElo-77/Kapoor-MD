import os
from pathlib import Path

def main(
    fasta_dir: Path,
    esm_out_dir: Path,
    out_dir: Path,
): 
    out_fasta = out_dir / "top_seqs.fa"   
    with out_fasta.open("w") as w:
        for root,  dirs, _ in esm_out_dir.walk():
            for dir in dirs: 
                seq_id = dir

                fasta_file = fasta_dir / f"{seq_id}.fa"
                if fasta_file.exists():
                    with fasta_file.open("r") as r:
                        w.write(r.read())
        
if __name__ == "__main__":

    FASTA_DIR = Path("lanm_pipeline/mpnn/outputs/fastas")
    ESM_OUT_DIR = Path("lanm_pipeline/af_esm/outputs/residue_preds/")
    OUT_DIR = Path("lanm_pipeline/clusters/inputs/")

    main(FASTA_DIR, ESM_OUT_DIR, OUT_DIR)
        
