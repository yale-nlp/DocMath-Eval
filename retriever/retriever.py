import json, time
import numpy as np
import argparse
from tqdm import tqdm
import warnings
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset
from typing import List
import os
import pickle
import src.index
from src.index import index_encoded_data
from openai import OpenAI, AsyncOpenAI, OpenAIError
import asyncio
import openai
import re
from rank_bm25 import BM25Okapi
from utils import bm25_utils
import tiktoken

warnings.filterwarnings("ignore")

def truncate_text_tokens(text, max_tokens=8000):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    trunc_encode = encoding.encode(text)[:max_tokens]
    new_text = encoding.decode(trunc_encode)
    return new_text

def hf_retriever_encode(model_name, texts: List[str], client, batch_size: int = 32) -> List[np.array]:
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda")
    model.to(device)

    batches = []
    embedding_list = []

    for i in range(0, len(texts), batch_size):
        batches.append(texts[i:i+batch_size])
    
    for batch in batches:
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        
        embeddings = model(**inputs)
        embedding_list.append(embeddings[0][:, 0, :].detach().cpu().numpy())

    return np.concatenate(embedding_list, axis=0)

def gpt_retriever_encode(model_name, texts, client):
    embeddings = []
    batch_size = 2000 # https://github.com/openai/openai-python/issues/519
    text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    for text_batch in text_batches:
        for _ in range(5):
            try:
                responses = client.embeddings.create(input=text_batch, model=model_name).data
                break
            except openai.BadRequestError as e:
                print(f"BadRequestError: {e}")
                return None
            except openai.RateLimitError as e:
                print(f"RateLimitError: {e}")
                time.sleep(10)
        
        for response in responses:
            embeddings.append(response.embedding)
    
    return np.array(embeddings)
    
projection_dict = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "facebook/contriever-msmarco": 768
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/contriever-msmarco", choices=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002", "facebook/contriever-msmarco", "bm25"])
    parser.add_argument("--dataset_name", type=str, default="yilunzhao/DocMath-Eval")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="retrieved_output")
    parser.add_argument("--n_subquantizers", type=int, default=0, help="Number of subquantizer used for vector quantization, if 0 flat index is used")
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")

    args = parser.parse_args()

    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base
    os.environ["OPENAI_API_KEY"] = args.api_key
    client = OpenAI()
    OpenAI.api_key = os.getenv('OPENAI_API_KEY')

    encoding = tiktoken.get_encoding("cl100k_base")
    os.makedirs(args.output_dir, exist_ok=True)
    subset_output_dir = os.path.join(args.output_dir, args.subset)
    os.makedirs(subset_output_dir, exist_ok=True)
    
    # load data
    data = load_dataset(args.dataset_name, split=args.subset)
    

    query_text_list = []
    top_k_results = []
    
    model_name = args.model_name.split("/")[-1]
    
    if args.top_k == -1:
        subdir = "all"
    else:
        subdir = f"top-{args.top_k}"
        
    os.makedirs(os.path.join(subset_output_dir, subdir), exist_ok=True)
    output_path = os.path.join(subset_output_dir, subdir, f"{model_name}.json")

    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}")
        exit()
    if args.model_name == "bm25":
        for i in range(len(data)):
            query_text_list.append(bm25_utils.process_text(data[i]['question']))
        for i in tqdm(range(len(data))):
            # load context
            context_list = data[i]['paragraphs']
            context_list = [bm25_utils.process_text(context) for context in context_list]
            
            bm25 = BM25Okapi(context_list)
            
            doc_scores = bm25.get_scores(query_text_list[i])
            
            # Get indices of top k elements
            top_k_indices = np.argsort(doc_scores)[::-1][:args.top_k]

            # Get values corresponding to top k indices
            top_k_values = doc_scores[top_k_indices]
            
            top_k_results.append((top_k_indices, top_k_values))
    else:
        # embed query
        encode_fn = gpt_retriever_encode if args.model_name in ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"] else hf_retriever_encode

        projection_size = projection_dict[args.model_name]
        for i in range(len(data)):
            query_text_list.append(data[i]['question'])

        query_embeddings = encode_fn(args.model_name, query_text_list, client)

        for i in tqdm(range(len(data))):
            # create index
            index = src.index.Indexer(projection_size, args.n_subquantizers, args.n_bits)
            
            # load context
            context_list = [truncate_text_tokens(i) if i.strip() else "empty string" for i in data[i]['paragraphs']]

            # index context
            context_embeddings = encode_fn(args.model_name, context_list, client)

            assert len(context_list) == len(context_embeddings)
            ids = list(range(len(context_list)))

            index_encoded_data(index, ids, context_embeddings, args.indexing_batch_size)

            k = len(context_list) if args.top_k == -1 else args.top_k
            # search top k paragraphs for each query
            top_k_paragraphs_and_scores = index.search_knn(query_embeddings[i].reshape(1, -1), k)
            
            top_k_results.append(top_k_paragraphs_and_scores[0])

    # store retrieved information
    output_data = []
    for i in range(len(data)):
        example_id = data[i]['question_id']
        retrieved_paragraphs = [(int(paragraph_id), float(score)) for paragraph_id, score in zip(*top_k_results[i])]
        output_data.append({
            "question_id": example_id,
            "retrieved_paragraphs": sorted(retrieved_paragraphs, key=lambda x: x[0])
        })

    json.dump(output_data, open(output_path, "w"), indent=4)