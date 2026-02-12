#!/bin/bash

while getopts "n:t:s:" opt; do
        case "$opt" in
                n) num_seqs="$OPTARG" ;;
                t) temp="$OPTARG" ;;
                s) score_cutoff="$OPTARG" ;;
        esac
done

python3 -m lanm_pipeline.mpnn.src.generate_seqs --num_seqs "$num_seqs" --temperature "$temp" --score_cutoff "$score_cutoff"
python3 -m lanm_pipeline.mpnn.outputs.seq2fastas 