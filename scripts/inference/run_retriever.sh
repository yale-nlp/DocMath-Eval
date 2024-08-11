#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

models=(
    'facebook/contriever-msmarco'
    'bm25'
    'text-embedding-3-small'
    'text-embedding-3-large'
)

sets=(
  "complong_testmini"
  "complong_test"
)

top_ks=(
  3
  5
  10
)

api_base="TODO"
api_key="TODO"

for model in "${models[@]}"; do
  for set in "${sets[@]}"; do
    echo "Running embedding for model: $model on $set set with top_k: $k"

    python retriever/retriever.py \
        --model_name "$model" \
        --subset "$set" \
        --top_k -1 \
        --api_base "$api_base" \
        --api_key "$api_key"
  done
done

for model in "${models[@]}"; do
  for set in "${sets[@]}"; do
    for k in "${top_ks[@]}"; do
      echo "Running embedding for model: $model on $set set with top_k: $k"

      model_name=$(echo $model | awk -F'/' '{print $NF}')
      input_file="retrieved_output/$set/all/${model_name}.json"
      output_dir="retrieved_output/$set/top_$k"
      python retriever/get_top_n.py \
          --input_file "$input_file" \
          --output_dir "$output_dir" \
          --top_k "$k"
    done
  done
done

