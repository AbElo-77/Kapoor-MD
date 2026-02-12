#!/bin/bash

#SBATCH --job-name=lanm_pipeline-1
#SBATCH --job-name=lanm_pipeline-1
#SBATCH  -A klab-biophysics
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH –ntasks-per-node=6
#SBATCH --partition=mb-h100
#SBATCH --mem=48G
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --output=lanm_pipeline-1.out
#SBATCH --error=lanm_pipeline-1.err
#SBATCH --time=168:00:00

export XLA_FLAGS="--xla_gpu_enable_bf16_to_fp32_rewriter=true"
export JAX_DEFAULT_DTYPE_BITS=32

while getopts "n:t:s:c:S:i:p:C:" opt; do
        case "$opt" in
                n) num_seqs="$OPTARG" ;;
                t) temp="$OPTARG" ;;
                s) score_cutoff="$OPTARG" ;;
                c) core_thresh="$OPTARG" ;;
                S) plddt_cutoff="$OPTARG" ;;
                i) ion="$OPTARG" ;;
                p) probability_cutoff="$OPTARG" ;;
                C) clusters="$OPTARG" ;;
        esac
done

bash lanm_pipeline/mpnn/run_mpnn.sh -n "$num_seqs" -t "$temp" -s "$score_cutoff"
bash lanm_pipeline/af_esm/run_alphafold.sh -c "$core_thresh" -S "$plddt_cutoff"
bash lanm_pipeline/af_esm/run_esmbind.sh -i "$ion" -p "$probability_cutoff"
bash lanm_pipeline/clusters/run_clustering.sh -C "$clusters"

