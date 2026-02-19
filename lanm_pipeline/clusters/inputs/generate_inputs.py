import os, json
from pathlib import Path

def main(
    fasta_dir: Path,
    af_out_dir: Path, 
    esm_out_dir: Path,
    out_dir: Path,
    out_metrics: Path
): 
    out_fasta = out_dir / "top_seqs.fa"   
    with out_fasta.open("w") as w:

        plddts = []
        rmsds = []
        num_abv = []
        probs = []
        for root,  dirs, _ in os.walk(esm_out_dir):
            for dir in dirs: 
                seq_id = dir

                metrics = af_out_dir / seq_id / "metrics.json"
                ca_preds = esm_out_dir / seq_id / "CA_preds.json"

                with open(metrics, 'r') as file:
                    data = json.load(file)
                
                if data["mean_plddt"] < 86 or data["rmsd_plddt_abv_80"] > 2.5 or (data["num_plddt_abv_80"] / data["length"]) < 0.96: 
                    continue

                plddts.append(data["mean_plddt"])
                rmsds.append(data["rmsd_plddt_abv_80"])
                num_abv.append(data["num_plddt_abv_80"])

                json_path = out_metrics / seq_id / "af_metrics.json"
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w") as jf:
                        json.dump(data, jf, indent=4)

                with open(ca_preds, 'r') as file:
                    data = json.load(file)

                if data["avg_score"] < 0.97: 
                    continue

                probs.append(data["avg_score"])

                json_path = out_metrics / seq_id / "esm_metrics.json"
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w") as jf:
                        json.dump(data, jf, indent=4)

                fasta_file = fasta_dir / f"{seq_id}.fa"
                if fasta_file.exists():
                    with fasta_file.open("r") as r:
                        w.write(r.read())

    out_csv = out_metrics / "overall_metrics.csv"
    with out_csv.open("w") as f:
        f.write("avg_plddt, avg_rmsd, avg_num_abv_80, avg_metal_binding_probability\n")
        f.write(
            f"{(sum(plddts) / len(plddts))},"
            f"{(sum(rmsds) / len(rmsds))},"
            f"{(sum(num_abv) / len(num_abv))},"
            f"{(sum(probs) / len(probs))}\n")

if __name__ == "__main__":

    FASTA_DIR = Path("lanm_pipeline/mpnn/outputs/fastas")
    AF_OUT_DIR = Path("lanm_pipeline/af_esm/outputs/structure_preds/")
    ESM_OUT_DIR = Path("lanm_pipeline/af_esm/outputs/residue_preds/")
    OUT_DIR = Path("lanm_pipeline/clusters/inputs/")
    OUT_METRICS = Path("lanm_pipeline/clusters/inputs/metrics")

    main(FASTA_DIR, AF_OUT_DIR, ESM_OUT_DIR, OUT_DIR, OUT_METRICS)