#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:$(pwd)

HF_API_TOKEN="TODO"
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('$HF_API_TOKEN')"

models=(
  "microsoft/Phi-3-medium-128k-instruct"
  "microsoft/Phi-3-mini-128k-instruct"

  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "meta-llama/Meta-Llama-3.1-70B-Instruct"

  'mistralai/Mistral-Nemo-Instruct-2407'

  'THUDM/glm-4-9b-chat'
)

sets=(
  "complong_testmini"
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