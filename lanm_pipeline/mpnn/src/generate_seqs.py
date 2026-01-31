import argparse
import os
import copy
import torch
import numpy as np

from lanm_pipeline.mpnn.src.ProteinMPNN.protein_mpnn_utils import (
    ProteinMPNN,
    parse_PDB,
    StructureDatasetPDB,
    tied_featurize,
    _S_to_seq,
    _scores
)

# -----------------------------------------------------------------


MODEL_WEIGHTS = "lanm_pipeline/mpnn/src/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
BATCH_SIZE = 1               

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DESIGNED_CHAINS = ["A", "B", "C", "D"]
FIXED_CHAINS = []

# -----------------------------------------------------------------

def main(PDB_PATH: str, OUT_DIR: str, NUM_SEQS: int, TEMPERATURE: float, SCORE_CUTOFF: float):
    os.makedirs(OUT_DIR, exist_ok=True)

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)

    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # -----------------------------------------------------------------

    pdb_dict_list = parse_PDB(PDB_PATH, input_chain_list=(DESIGNED_CHAINS + FIXED_CHAINS))
    dataset = StructureDatasetPDB(pdb_dict_list, max_length=20_000)

    chain_id_dict = {
        pdb_dict_list[0]["name"]: (DESIGNED_CHAINS, FIXED_CHAINS)
    }

    protein = dataset[0]

    fasta_path = os.path.join(OUT_DIR, "hans_mpnn.fasta")
    fasta = open(fasta_path, "w")

    generated = 0
    batch_id = 0

    # -----------------------------------------------------------------

    pssm_multi, pssm_threshold, pssm_log_odds_flag, pssm_bias_flag = 0, 0, 0, 0
    omit_AAs_list = 'X'

    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

    chain_id_dict = None
    fixed_positions_dict = None
    pssm_dict = None
    omit_AA_dict = None

    bias_AA_dict = None
    tied_positions_dict = None
    bias_by_res_dict = None
    bias_AAs_np = np.zeros(len(alphabet))

    # -----------------------------------------------------------------

    with torch.no_grad():
        while generated < NUM_SEQS:

            batch = [copy.deepcopy(protein) for _ in range(BATCH_SIZE)]

            (X, S, mask, lengths, chain_M, chain_encoding_all, 
            chain_list_list, visible_list_list, masked_list_list, 
            masked_chain_length_list_list, chain_M_pos, omit_AA_mask, 
            residue_idx, dihedral_mask, tied_pos_list_of_lists_list, 
            pssm_coef, pssm_bias, pssm_log_odds_all, 
            bias_by_res_all, tied_beta) = tied_featurize(batch, 
                                                        DEVICE, 
                                                        chain_id_dict, 
                                                        fixed_positions_dict, 
                                                        omit_AA_dict, 
                                                        tied_positions_dict, 
                                                        pssm_dict, 
                                                        bias_by_res_dict)
            
            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float()
            name_ = batch[0]['name']
            
            randn = torch.randn(chain_M.shape, device=DEVICE)

            sample = model.sample(
                X,
                randn,
                S,
                chain_M,
                chain_encoding_all,
                residue_idx,
                mask=mask,
                chain_M_pos=chain_M_pos,
                omit_AAs_np=omit_AAs_np,
                bias_AAs_np=bias_AAs_np,
                bias_by_res=bias_by_res_all,
                omit_AA_mask=omit_AA_mask,
                temperature=TEMPERATURE
            )

            S_sample = sample["S"]

            for i in range(BATCH_SIZE):
                if generated >= NUM_SEQS:
                    break
                log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn, use_input_decoding_order=True, decoding_order=sample["decoding_order"])
                scores = _scores(S_sample[i], log_probs, mask[i]*chain_M[i]*chain_M_pos[i])

                if scores > SCORE_CUTOFF:
                    print(f"[MPNN] Dropped sequence with score {scores:.2f}")
                    continue

                seq = _S_to_seq(S_sample[i], chain_M[i])
                
                fasta.write(f">seq_{generated}\n{seq}\n")
                generated += 1

            batch_id += 1
            if batch_id % 100 == 0:
                print(f"[MPNN] Generated {generated}/{NUM_SEQS}")

    fasta.close()

if __name__ == "__main__":

    PDB_PATH = "lanm_pipeline/mpnn/src/pdb/3CLN.pdb"
    OUT_DIR  = "lanm_pipeline/mpnn/outputs/seqs"
    NUM_SEQS = 50
    TEMPERATURE = 0.1
    SCORE_CUTOFF = 1

    argparser = argparse.ArgumentParser(description="Generate sequences with ProteinMPNN")
    argparser.add_argument("--pdb_path", type=str, default=PDB_PATH, help="Path to input PDB file")
    argparser.add_argument("--out_dir", type=str, default=OUT_DIR, help="Output directory for generated sequences")
    argparser.add_argument("--num_seqs", type=int, default=NUM_SEQS, help="Number of sequences to generate")
    argparser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature")
    argparser.add_argument("--score_cutoff", type=float, default=SCORE_CUTOFF, help="Score cutoff for generated sequences")
    args = argparser.parse_args()

    PDB_PATH = args.pdb_path
    OUT_DIR = args.out_dir
    NUM_SEQS = args.num_seqs
    TEMPERATURE = args.temperature
    SCORE_CUTOFF = args.score_cutoff

    main(PDB_PATH, OUT_DIR, NUM_SEQS, TEMPERATURE, SCORE_CUTOFF)