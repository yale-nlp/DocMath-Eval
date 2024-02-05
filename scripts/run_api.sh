#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)
models=(
    'gpt-4-turbo'
)

sets=(
  "compshort_testmini"
  "compshort_test"
  "simpshort_testmini"
  "simpshort_test"
  "simplong_testmini"
  "simplong_test"
)

api_base="TODO"
api_key="TODO"

for model in "${models[@]}"; do
  for set in "${sets[@]}"; do
    echo "Running inference for model: $model on $set set"
    requests_per_minute=100

    python run_llm.py \
        --model_name "$model" \
        --prompt_type cot \
        --output_dir outputs/llm_outputs \
        --subset "$set" \
        --max_tokens 512 \
        --api \
        --api_base "$api_base" \
        --api_key "$api_key" \
        --requests_per_minute "$requests_per_minute"
      
    python run_llm.py \
        --model_name "$model" \
        --prompt_type pot \
        --output_dir outputs/llm_outputs \
        --subset "$set" \
        --max_tokens 512 \
        --api \
        --api_base "$api_base" \
        --api_key "$api_key" \
        --requests_per_minute "$requests_per_minute"
  done
done


# RAG Setting for Complong subset (top-10 evidence retrieved by text-embedding-3-large)
sets=(
  "complong_testmini"
  "complong_test"
)

for model in "${models[@]}"; do
  for set in "${sets[@]}"; do
    echo "Running inference for model: $model on $set set"
    requests_per_minute=100

    python run_llm.py \
        --model_name "$model" \
        --gpu_memory_utilization 0.9 \
        --prompt_type cot \
        --output_dir outputs/llm_outputs \
        --retriever \
        --subset "$set" \
        --max_tokens 512 \
        --api \
        --api_base "$api_base" \
        --api_key "$api_key" \
        --requests_per_minute "$requests_per_minute"
      
    python run_llm.py \
        --model_name "$model" \
        --gpu_memory_utilization 0.9 \
        --prompt_type pot \
        --output_dir outputs/llm_outputs \
        --retriever \
        --subset "$set" \
        --max_tokens 512 \
        --api \
        --api_base "$api_base" \
        --api_key "$api_key" \
        --requests_per_minute "$requests_per_minute"
  done
done