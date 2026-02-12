#!/bin/bash

while getopts "i:p:" opt; do
        case "$opt" in
                i) ion="$OPTARG" ;;
                p) probability_cutoff="$OPTARG" ;;
        esac
done

python3 -m lanm_pipeline.af_esm.src.residue_predictions --ion "$ion" --probability_cutoff "$probability_cutoff"