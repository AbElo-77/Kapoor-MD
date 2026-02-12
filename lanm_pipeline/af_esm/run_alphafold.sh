#!/bin/bash

while getopts "c:S:" opt; do
        case "$opt" in
                c) core_thresh="$OPTARG" ;;
                S) plddt_cutoff="$OPTARG" ;;
        esac
done

python3 -m lanm_pipeline.af_esm.src.generate_pdbs --threshold "$core_thresh" --plddt_cutoff "$plddt_cutoff"
python3 -m lanm_pipeline.af_esm.outputs.clean_a3m