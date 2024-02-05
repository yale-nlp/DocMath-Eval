from tqdm import tqdm
import tiktoken
import re

def count_tokens(input_str):
    enc = tiktoken.encoding_for_model("gpt-4")
    input_token_len = len(enc.encode(input_str))
    return input_token_len


def get_context(index, example, subset, retrieve_data):
    context = ''

    retrieval_evidence = [i for i, _ in retrieve_data[index]['retrieved_paragraphs']]

    for i in retrieval_evidence:
        context += example['paragraphs'][i] + '\n'
    return context


def prepare_pot_model_input(index, example, subset, retriever, max_context_tokens):
    system_input = """
You are a financial expert, you are supposed to generate a Python program to answer the given question based on the provided financial document context. The returned value of the program is supposed to be the answer. 
```python
def solution():
    # Define variables name and value based on the given context
    guarantees = 210
    total_exposure = 716

    #Do math calculation to get the answer
    answer = (guarantees / total_exposure) * 100

    # return answer
    return answer
```
"""
    program_prefix_input = '''Please generate a Python program to answer the given question. The format of the program should be the following:
```python
def solution():
    # Define variables name and value based on the given context
    ...
    # Do math calculation to get the answer
    ...
    # return answer
    return answer
```

Continue the program to answer the question. The returned value of the program is supposed to be the answer:
```python
def solution():
    # Define variables name and value based on the given context
'''

    if 'complong' in subset and retriever is not None:
        input_context = get_context(index, example, subset, retriever)
    else:
        input_context = '\n'.join(example['paragraphs'])
    
    user_input = input_context + "\n\n"
    user_input += f"Question: {example['question']}\n\n"
    user_input += program_prefix_input
    return system_input, user_input

def prepare_cot_model_input(index, example, subset, retriever, max_context_tokens):
    system_input = "You are a financial expert, you are supposed to answer the given question based on the provided financial document context. You need to first think through the problem step by step, documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as 'Therefore, the answer is {final answer}'. The final answer should be a numeric value."
    user_input = ''
    if 'complong' in subset and retriever is not None:
        input_context = get_context(index, example, subset, retriever)
    else:
        input_context = '\n'.join(example['paragraphs'])
        

    user_input += input_context + "\n\n"
    user_input += f"Question: {example['question']}\n\nLet's think step by step to answer the given question."
    
    return system_input, user_input
    
def prepare_model_inputs(qa_data, subset, prompt_type, model_name, api_based, tokenizer=None, retrieval_data=None, max_context_tokens=None):
    model_inputs = []
    for index in tqdm(range(len(qa_data))):
        if prompt_type == "pot":
            system_input, user_input = prepare_pot_model_input(index, qa_data[index], subset, retrieval_data, max_context_tokens)
        else:
            system_input, user_input = prepare_cot_model_input(index, qa_data[index], subset, retrieval_data, max_context_tokens)
            
        models_without_system = ("gemma", "OLMo", "Mistral", "Mixtral", "starcoder2")
        if any(model in model_name for model in models_without_system):
            model_input = [
                {"role": "user", "content": system_input + "\n" + user_input}
            ]
        else:
            model_input = [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input}
            ]
        if not api_based:
            model_input = tokenizer.apply_chat_template(model_input, tokenize=False)
        
        model_inputs.append(model_input)
    return model_inputs