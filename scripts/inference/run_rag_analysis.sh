#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:$(pwd)

sets=(
  "complong_testmini"
)

top_ks=(
  3
  5
  10
)
retrievers=(
    'contriever-msmarco'
    'bm25'
    'text-embedding-3-small'
    'text-embedding-3-large'
)

api_base="TODO"
api_key="TODO"

# GPT-4o
for retrieval_model in "${retrievers[@]}"; do
    for k in "${top_ks[@]}"; do
        echo "Running embedding for model: $model on $set set with top_k: $k"
        python run_llm.py \
            --model_name gpt-4o \
            --prompt_type cot \
            --output_dir rag_analysis_output \
            --retriever \
            --retriever_model_name "$retrieval_model" \
            --subset complong_testmini \
            --topk "$k" \
            --max_tokens 512 \
            --api \
            --api_base "$api_base" \
            --api_key "$api_key" \
            --requests_per_minute 200
    done
done

# Llama-3-70B-Instruct
for retrieval_model in "${retrievers[@]}"; do
    for k in "${top_ks[@]}"; do
        echo "Running embedding for model: $model on $set set with top_k: $k"
        python run_llm.py \
            --model_name meta-llama/Meta-Llama-3-70B-Instruct \
            --gpu_memory_utilization 0.9 \
            --prompt_type cot \
            --output_dir rag_analysis_output \
            --retriever \
            --retriever_model_name "$retrieval_model" \
            --subset complong_testmini \
            --topk "$k" \
            --max_tokens 512
    done
done