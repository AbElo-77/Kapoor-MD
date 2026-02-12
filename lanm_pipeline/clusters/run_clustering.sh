#!/bin/bash

while getopts "C:" opt; do
        case "$opt" in
                C) clusters="$OPTARG" ;;
        esac
done

python3 -m lanm_pipeline.clusters.inputs.generate_inputs
python3 -m lanm_pipeline.clusters.cluster --k "$clusters"