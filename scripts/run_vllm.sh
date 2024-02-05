#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:$(pwd)

HF_API_TOKEN="TODO"
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('$HF_API_TOKEN')"

models=(
  "meta-llama/Llama-2-70b-chat"
)

sets=(
  "compshort_testmini"
  "compshort_test"
  "simpshort_testmini"
  "simpshort_test"
  "simplong_testmini"
  "simplong_test"
)

for model in "${models[@]}"; do
  for set in "${sets[@]}"; do
    echo "Running inference for model: $model on $set"

    python run_llm.py \
        --model_name "$model" \
        --gpu_memory_utilization 0.9 \
        --prompt_type cot \
        --output_dir outputs/llm_outputs \
        --subset "$set" \
        --max_tokens 512
      
    python run_llm.py \
        --model_name "$model" \
        --gpu_memory_utilization 0.9 \
        --prompt_type pot \
        --output_dir outputs/llm_outputs \
        --subset "$set" \
        --max_tokens 512
  done
done

# RAG Setting for Complong subset (top-10 evidence retrieved by text-embedding-3-large)
sets=(
  "complong_testmini"
  "complong_test"
)

for model in "${models[@]}"; do
  for set in "${sets[@]}"; do
    echo "Running inference for model: $model on $set"
    

    python run_llm.py \
        --model_name "$model" \
        --gpu_memory_utilization 0.9 \
        --prompt_type cot \
        --output_dir outputs/llm_outputs \
        --retriever \
        --subset "$set" \
        --max_tokens 512
      
    python run_llm.py \
        --model_name "$model" \
        --gpu_memory_utilization 0.9 \
        --prompt_type pot \
        --output_dir outputs/llm_outputs \
        --retriever \
        --subset "$set" \
        --max_tokens 512
  done
done