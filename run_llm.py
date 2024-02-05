from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
import json
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse
from transformers import AutoTokenizer
import random
from typing import Union
import asyncio
import openai
from utils.openai_utils import *
from utils.model_input_utils import prepare_model_inputs


def process_single_example_raw_outputs(outputs):
    processed_outputs = []
    assert len(outputs.outputs) == 1
    processed_outputs.append(outputs.outputs[0].text)
    return processed_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    
    # dataset and output
    parser.add_argument("--dataset_name", type=str, default="yale-nlp/DocMath-Eval")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")

    # retriever setting for simplong and complong
    parser.add_argument("--retriever", action="store_true")
    parser.add_argument("--retriever_output_dir", type=str, default="retrieved_output")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--retriever_model_name", type=str, default="text-embedding-3-large", choices=["text-embedding-3-large", "text-embedding-3-small", "contriever-msmarco", "text-embedding-ada-002", "bm25"])
    parser.add_argument("--max_context_tokens", type=int, default=3500)
    
    # llm setting
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=int, default=1.0)
    parser.add_argument("--prompt_type", type=str, default="cot", choices=["pot", "cot"])
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_num_examples", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--quantization", type=str, default="")
    
    # api key
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--requests_per_minute", type=int, default=100)
    
    args = parser.parse_args()
    
    gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    qa_data = load_dataset(args.dataset_name, split=args.subset)
    
    if args.max_num_examples > 0:
        qa_data = qa_data.select(range(args.max_num_examples))
    
    suffix_model_name = args.model_name.split("/")[-1].replace(".", "_")
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.retriever:
        output_dir = os.path.join(args.output_dir, args.subset, f"raw_{args.prompt_type}_outputs")
        retrieval_data = None
        output_file = os.path.join(output_dir, f"{suffix_model_name}.json")
    else:
        if args.output_dir == "rag_analysis_output":
            output_file = os.path.join(args.output_dir, f"{suffix_model_name}-{args.retriever_model_name}-top_{args.topk}.json")
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(args.output_dir, f"{args.subset}-rag", f"raw_{args.prompt_type}_outputs")
            output_file = os.path.join(output_dir, f"{suffix_model_name}.json")

        retrieved_filepath = os.path.join(args.retriever_output_dir, args.subset, f"top_{args.topk}", f"{args.retriever_model_name}.json")
        if not os.path.exists(retrieved_filepath):
            raise FileNotFoundError(f"Retrieved file not found: {retrieved_filepath}")
        retrieval_data = json.load(open(retrieved_filepath, "r"))

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        exit()

    if not args.api:
        if args.quantization:
            llm = LLM(args.model_name,
                    tensor_parallel_size=gpu_count,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    trust_remote_code=True,
                    quantization=args.quantization)
        else:
            llm = LLM(args.model_name, 
                    tensor_parallel_size=gpu_count, 
                    dtype="half" if "gemma-2" not in args.model_name else "bfloat16", # https://github.com/vllm-project/vllm/issues/6177
                    swap_space=16, 
                    gpu_memory_utilization=args.gpu_memory_utilization, 
                    trust_remote_code=True)
        
        sampling_params = SamplingParams(temperature = args.temperature, 
                                        top_p = args.top_p, 
                                        max_tokens = args.max_tokens)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, verbose=False, trust_remote_code=True)
        tokenizer.use_default_system_prompt = True
        model_inputs = prepare_model_inputs(qa_data, args.subset, args.prompt_type, args.model_name, args.api, tokenizer, retrieval_data, args.max_context_tokens) 
        
        outputs = llm.generate(model_inputs, sampling_params)
        raw_outputs = [process_single_example_raw_outputs(output) for output in outputs]
    
    else:
        if args.api_base:
            os.environ["OPENAI_BASE_URL"] = args.api_base
        os.environ["OPENAI_API_KEY"] = args.api_key
        client = AsyncOpenAI()
        AsyncOpenAI.api_key = os.getenv('OPENAI_API_KEY')

        model_inputs = prepare_model_inputs(qa_data, args.subset, args.prompt_type, args.model_name, args.api, None, retrieval_data, args.max_context_tokens)
        model_name = args.model_name              

        raw_outputs = asyncio.run(generate_from_openai_chat_completion( 
                                                    client = client,
                                                    messages = model_inputs,
                                                    engine_name = args.model_name, 
                                                    temperature = args.temperature, 
                                                    top_p = args.top_p, 
                                                    max_tokens = args.max_tokens,
                                                    requests_per_minute = args.requests_per_minute,))
    
    
    output_data = []
    for raw_output, qa in zip(raw_outputs, qa_data):
        if type(raw_output) != list:
            qa["output"] = [raw_output]
        else:
            qa["output"] = raw_output
        del qa["paragraphs"]
        del qa["table_evidence"]
        del qa["paragraph_evidence"]
        output_data.append(qa)
        
    json.dump(output_data, open(output_file, "w"), indent=4, ensure_ascii=True)