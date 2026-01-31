import argparse
import os, re, json
from pathlib import Path

import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from alphafold.common import protein as alphafold_protein
from alphafold.common import residue_constants
from alphafold.data import pipeline, parsers
from alphafold.model import data, config
from alphafold.model import model as alphafold_model

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from Bio.PDB.Superimposer import Superimposer

# -----------------------------------------------------
PARAMS_DIR = Path("lanm_pipeline/af_esm/params")

NUM_RECYCLE = 12
MODEL_NAME  = "model_3_ptm"

# -----------------------------------------------------

def read_fasta_one(path: Path) -> tuple[str, str]:
    seq_lines = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            else:
                seq_lines.append(line)
    seq = "".join(seq_lines).upper()
    seq = re.sub("[^A-Z:/]", "", seq)
    if not seq:
        raise ValueError(f"Empty FASTA: {path}")
    return (Path(os.path.basename(path)).stem), seq

        
def generate_a3m(fasta_dir: Path, out_root: Path) -> str:
    fasta_out = out_root.parent / "sequences.fasta"

    with open(fasta_out, "w") as w:
        for fa in sorted(list(fasta_dir.rglob("*.fa"))):
            
            seq_id = fa.stem
            seq = "".join(
                line.strip() for line in fa.read_text().splitlines()
                if line and not line.startswith(">")
            ).upper()
            seq = "".join([c for c in seq if "A" <= c <= "Z"])
            if not seq:
                continue
            w.write(f">{seq_id}\n{seq}\n")
    try: 
        subprocess.run([
            "colabfold_batch", 
            str(fasta_out),
            str(out_root.parent / "a3m"), 
            "--msa-only"], 
            check=True, 
            capture_output=True)
    except subprocess.CalledProcessError as e:
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)


def chain_break(residue_index: np.ndarray, Ls: list[int], gap: int = 32) -> np.ndarray:
    out = residue_index.copy()
    pos = 0
    for L in Ls[:-1]:
        pos += L
        out[pos:] += gap
    return out

def get_plddt(outputs):
    logits = outputs["predicted_lddt"]["logits"]
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = jnp.arange(0.5 * bin_width, 1.0, bin_width)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * bin_centers[None, :], axis=-1)

def get_pae(outputs):
    pae_data = outputs.get("predicted_aligned_error")
    if pae_data is None:
        return None
    if isinstance(pae_data, dict):
        logits = pae_data["logits"]
        breaks = pae_data["breaks"]
    else:
        logits = pae_data
        breaks = jnp.linspace(0., 31.0, logits.shape[-1] - 1) 
    
    prob = jax.nn.softmax(logits, axis=-1)
    step = breaks[1] - breaks[0]
    bin_centers = breaks + step / 2
    bin_centers = jnp.append(bin_centers, bin_centers[-1] + step)
    return jnp.sum(prob * bin_centers[None, None, :], axis=-1)

def _ca_by_residue(struct):
    out = {}
    for model in struct:
        for chain in model:
            for res in chain:
                if not is_aa(res, standard=False):
                    continue
                if "CA" not in res:
                    continue
                resseq = res.get_id()[1]
                icode  = res.get_id()[2].strip()
                out[(chain.id, resseq, icode)] = res["CA"]
        break 
    return out

def get_rmsd(pred_pdb, ref_pdb):
    parser = PDBParser(QUIET=True)
    pred = parser.get_structure("pred", pred_pdb)
    ref  = parser.get_structure("ref",  ref_pdb)

    pred_map = _ca_by_residue(pred)
    ref_map  = _ca_by_residue(ref)

    keys = sorted(set(pred_map) & set(ref_map))
    if len(keys) < 10:
        raise ValueError(f"Too few matched residues: {len(keys)}")

    pred_atoms = [pred_map[k] for k in keys]
    ref_atoms  = [ref_map[k] for k in keys]

    sup = Superimposer()
    sup.set_atoms(ref_atoms, pred_atoms)
    sup.apply(pred.get_atoms())
    return sup.rms

def get_core_rmsd(
    pred_pdb: str,
    ref_pdb: str,
    plddt: np.ndarray,
    plddt_thresh: float = 80.0,
    min_residues: int = 20,
):
    
    parser = PDBParser(QUIET=True)
    pred = parser.get_structure("pred", pred_pdb)
    ref  = parser.get_structure("ref",  ref_pdb)

    def get_ca_list(struct):
        cas = []
        for model in struct:
            for chain in model:
                for res in chain:
                    if is_aa(res, standard=False) and "CA" in res:
                        cas.append(res["CA"])
            break
        return cas

    pred_ca = get_ca_list(pred)
    ref_ca  = get_ca_list(ref)

    keep = np.where(plddt >= plddt_thresh)[0]
    if len(keep) < min_residues:
        raise ValueError(
            f"Too few confident residues for core RMSD: {len(keep)}"
        )

    pred_atoms = [pred_ca[i] for i in keep]
    ref_atoms  = [ref_ca[i]  for i in keep]

    sup = Superimposer()
    sup.set_atoms(ref_atoms, pred_atoms)
    sup.apply(pred.get_atoms())

    return sup.rms, len(keep)

def save_pdb(outs, filename: Path):
    p = {
        "residue_index": outs["residue_idx"] + 1,
        "aatype": outs["seq"],
        "atom_positions": outs["final_atom_positions"],
        "atom_mask": outs["final_atom_mask"],
        "plddt": get_plddt(outs),
        "chain_index": np.zeros(outs["seq"].shape[0], dtype=np.int32), 
    }
    p = jax.tree_map(lambda x: x[: outs["length"]], p)
    b_factors = 100.0 * p.pop("plddt")[:, None] * p["atom_mask"]
    prot = alphafold_protein.Protein(**p, b_factors=b_factors)
    pdb_lines = alphafold_protein.to_pdb(prot)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(pdb_lines)

def setup_model(max_len: int, a3m_path: Path = None):

    cfg = config.model_config(MODEL_NAME)

    cfg.model.return_representations = True 

    cfg.model.num_recycle = NUM_RECYCLE
    cfg.data.common.num_recycle = NUM_RECYCLE
    cfg.data.eval.max_msa_clusters = 512
    cfg.data.common.max_extra_msa = 1024  
    cfg.data.eval.masked_msa_replace_fraction = 0
    cfg.model.global_config.subbatch_size = None

    model_param = data.get_model_haiku_params(model_name=MODEL_NAME, data_dir=str(PARAMS_DIR.parent))
    model_runner = alphafold_model.RunModel(cfg, model_param)

    clean_lines = []
    for line in a3m_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not "\x00" in line:
                clean_lines.append(line)
        
    msa_str = "\n".join(clean_lines)
    msa = parsers.parse_a3m(msa_str)

    feature_dict = {
        **pipeline.make_sequence_features(sequence=msa.sequences[0], description="none", num_res=len(msa.sequences[0])),
        **pipeline.make_msa_features(msas=[msa]),
    }
    inputs = model_runner.process_features(feature_dict, random_seed=0)

    def runner(I, params):
        inputs = I["inputs"]
        inputs.update(I["prev"])

        L = inputs["residue_index"].shape[-1]
        mask_1d = (jnp.arange(L) < I["length"])

        seq_int = I["seq"]
        seq_int = jnp.where(mask_1d, seq_int, 0).astype(jnp.int32)


        if "aatype" in inputs:
            inputs["aatype"] = jnp.array(seq_int.reshape(1, -1))

        if "msa" in inputs:
            if inputs["msa"].ndim == 2:
                inputs["msa"] = inputs["msa"].at[0, :].set(seq_int)
            elif inputs["msa"].ndim == 3:
                pass

        if "msa_feat" in inputs:
            seq_oh = jax.nn.one_hot(seq_int, 20).astype(inputs["msa_feat"].dtype)
            seq_oh = seq_oh[None, None, :, :]
            mf = jnp.zeros_like(inputs["msa_feat"])
            mf = mf.at[..., 0:20].set(seq_oh)
            if mf.shape[-1] >= 45:
                mf = mf.at[..., 25:45].set(seq_oh)
            inputs["msa_feat"] = mf

        if "target_feat" in inputs:
            tf = jnp.zeros_like(inputs["target_feat"])
            if tf.shape[-1] >= 21:
                tf = tf.at[..., 1:21].set(jax.nn.one_hot(seq_int, 20).astype(tf.dtype))
            inputs["target_feat"] = tf

        key = jax.random.PRNGKey(0)
        outputs = model_runner.apply(params, key, inputs)

        prev_outputs = outputs["prev"]
        prev = {
            "init_msa_first_row": prev_outputs["prev_msa_first_row"][None],
            "init_pair":          prev_outputs["prev_pair"][None],
            "init_pos":           prev_outputs["prev_pos"][None],
        }

        aux = {
            "final_atom_positions": outputs["structure_module"]["final_atom_positions"],
            "final_atom_mask": outputs["structure_module"]["final_atom_mask"],
            "length": I["length"],
            "seq": seq_int,
            "prev": prev,
            "residue_idx": (inputs["residue_index"][0] if inputs["residue_index"].ndim == 2 else inputs["residue_index"]),
            "predicted_lddt": outputs["predicted_lddt"],
            "predicted_aligned_error": outputs["predicted_aligned_error"]
        }
        return aux

    return jax.jit(runner), model_param, {"inputs": inputs, "length": max_len}

# -----------------------------------------------------

def main(INPUT_DIR: Path, PDB_PATH: Path, OUT_ROOT: Path, CUT_OFF: float, CORE_THRESHOLD: float):
    fastas = sorted(INPUT_DIR.rglob("*.fa")) + sorted(INPUT_DIR.rglob("*.fasta"))
    if not fastas:
        raise SystemExit(f"No Directory Found: {INPUT_DIR}")
    
    # generate_a3m(INPUT_DIR, OUT_ROOT)
    runner = None
    params = None
    I = None

    for idx, fasta_path in enumerate(fastas):
        fasta_id, ori_sequence = read_fasta_one(fasta_path)

        Ls = [len(s) for s in ori_sequence.replace(":", "/").split("/")]
        sequence = re.sub("[^A-Z]", "", ori_sequence)
        
        length = len(sequence)
        max_len = length

        msa_path = OUT_ROOT.parent / "a3m" / f"{fasta_id}.a3m"

        runner, params, I = setup_model(length, msa_path)
        seq_int = np.array([residue_constants.restype_order.get(aa, 0) for aa in sequence], dtype=np.int32)
        seq_int = np.pad(seq_int, [0, max_len - length], constant_values=-1)

        res_idx = chain_break(np.arange(max_len, dtype=np.int32), Ls, gap=32)
        I["inputs"]["residue_index"] = jnp.asarray(res_idx)[None, :] 

        I.update({"seq": jnp.asarray(seq_int), "length": int(length)})

        """
        The hardcoded numbers here are specific to the alphafold model being used. They are taken 
        from lines 449-451 within the modules_multimer.py file in the alphafold package.
        """
        I["prev"] = {
            "init_msa_first_row": np.zeros([1, max_len, 256], dtype=np.float32),
            "init_pair": np.zeros([1, max_len, max_len, 128], dtype=np.float32),
            "init_pos": np.zeros([1, max_len, 37, 3], dtype=np.float32),
        }

        O = None
        for _ in range(NUM_RECYCLE + 1):
            O = runner(I, params)
            O = jax.tree_map(lambda x: np.asarray(x), O)
            I["prev"] = O["prev"]

        plddt = get_plddt(O)[:length] * 100.0
        pae = get_pae(O)[:length, :length]

        if float(plddt.mean()) < CUT_OFF: 
            continue

        out_dir = OUT_ROOT / fasta_id
        pdb_file = out_dir / "predicted.pdb"
        save_pdb(O, pdb_file)

        rmsd_plddt, len_resd = get_core_rmsd(pdb_file, PDB_PATH, plddt, CORE_THRESHOLD)

        metrics = {
            "id": fasta_id,
            "fasta_path": str(fasta_path),
            "length": int(length),
            "mean_plddt": float(plddt.mean()),
            "num_plddt_abv_80": len_resd, 
            "min_plddt": float(plddt.min()),
            "mean_pae": float(pae.mean()),
            "rmsd": get_rmsd(pdb_file, PDB_PATH),
            "rmsd_plddt_abv_80": rmsd_plddt,
            "ref_pdb": str(PDB_PATH),
            "model": MODEL_NAME,
            "num_recycle": int(NUM_RECYCLE),
            "max_len": int(max_len),
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)

# -----------------------------------------------------

if __name__ == "__main__":
    INPUT_DIR = Path("lanm_pipeline/mpnn/outputs/fastas/")
    PDB_PATH  = Path("lanm_pipeline/mpnn/src/pdb/3CLN.pdb")
    OUT_ROOT  = Path("lanm_pipeline/af_esm/outputs/structure_preds/")
    CORE_THRESHOLD = 80.0
    CUT_OFF = 75.0 

    argparser = argparse.ArgumentParser(description="Generate ESM protein embeddings from FASTA file")
    argparser.add_argument("--fasta_dir", default=INPUT_DIR, help="Path to input FASTA file")
    argparser.add_argument("--pdb_path", default=PDB_PATH, help="Path to reference PDB file")
    argparser.add_argument("--output_dir", default=OUT_ROOT, help="Path to output directory")
    argparser.add_argument("--threshold", default=CORE_THRESHOLD, help="Minimum pLDDT for residues in core RMSD calculation.")
    argparser.add_argument("--plddt_cutoff", type=float, default=CUT_OFF, help="Minimum mean pLDDT to keep structure")
    args = argparser.parse_args()

    INPUT_DIR = Path(args.fasta_dir)
    PDB_PATH  = Path(args.pdb_path)
    OUT_ROOT = Path(args.output_dir)
    CUT_OFF = args.plddt_cutoff

    main(INPUT_DIR, PDB_PATH, OUT_ROOT, CUT_OFF, CORE_THRESHOLD)
