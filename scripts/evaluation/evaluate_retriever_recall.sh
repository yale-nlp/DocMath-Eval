#!/bin/bash

top_ks=(
  3
  5
  10
)

for k in "${top_ks[@]}"; do
    raw_dir="outputs/retrieved_output/complong_testmini/top_$k"
    for raw_file in "$raw_dir"/*; do
        filename=$(basename "$raw_file")
        python retriever/recall_evaluation.py \
            --input_file "$raw_file" \
            --ground_truth_file "data/complong_testmini.json"
    done
done