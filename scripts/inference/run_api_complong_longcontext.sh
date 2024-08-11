#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)
models=(    
    # Claude
    'claude-3-haiku-20240307'
    'claude-3-sonnet-20240229'
    'claude-3-5-sonnet-20240620'
    

    # GPT-4o
    'gpt-4o'
    'gpt-4o-mini'

    # Deepseek
    'deepseek-chat'
    'deepseek-coder'

    'gemini-1.5-flash'
    'gemini-1.5-pro'
)

sets=(
  "complong_testmini"
  "complong_test"
)

api_base="TODO"
api_key="TODO"

for model in "${models[@]}"; do
  for set in "${sets[@]}"; do
    echo "Running inference for model: $model on $set set"
    
    requests_per_minute=10
      

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
  done
done