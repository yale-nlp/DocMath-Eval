#!/bin/bash
api_base="TODO"
api_key="TODO"


raw_dir="outputs/rag_analysis_output/raw_outputs"
processed_dir="outputs/rag_analysis_output/processed_outputs"
result_file="outputs/rag_analysis_output/results.json"

# remove result file if it exists
if [ -f "$result_file" ]; then
    rm "$result_file"
fi

# Iterate over each file in the raw output directory
for raw_file in "$raw_dir"/*; do
    filename=$(basename "$raw_file")
    
    python evaluation.py \
        --prediction_path "$raw_file" \
        --evaluation_output_dir "$processed_dir" \
        --prompt_type cot \
        --ground_truth_file "data/complong_testmini.json" \
        --result_file "$result_file" \
        --api_base "$api_base" \
        --api_key "$api_key"

    echo "Finished evaluating $filename"
done
