## DocMath-Eval
The data and code for the paper [DocMath-Eval: Evaluating Math Reasoning Capabilities of LLMs in Understanding Long and Specialized Documents](https://arxiv.org/abs/2311.09805). 
**DocMath-Eval** is a comprehensive benchmark focused on numerical reasoning within specialized domains. It requires the model to comprehend long and specialized documents and perform numerical reasoning to answer the given question. 
**DocMath-Eval** includes **4,000 QA examples** across 4 subsets. These examples were collected by financial experts and feature detailed solution annotations in Python format.


## DocMath-Eval Dataset
All the data examples were divided into four subsets:

- **simpshort**, which is reannotated from [TAT-QA](https://aclanthology.org/2021.acl-long.254/) and [FinQA](https://aclanthology.org/2021.emnlp-main.300/), necessitates simple numerical reasoning over short document with one table
- **simplong**, which is reannotated from [MultiHiertt](https://aclanthology.org/2022.acl-long.454/), necessitates simple numerical reasoning over long document with multiple tables;
- **compshort**, which is reannotated from [TAT-HQA](https://aclanthology.org/2022.acl-long.5/), necessitates complex numerical reasoning over short document with one table;
- **complong**, which is annotated from scratch by our team, necessitates complex numerical reasoning over long document with multiple tables.

For each subset, we provide the *testmini* and *test* splits. 

You can download this dataset by the following command:

```python
from datasets import load_dataset

dataset = load_dataset("yale-nlp/DocMath-Eval")

# print the first example on the complong testmini set
print(dataset["complong-testmini"][0])
```

The dataset is provided in json format and contains the following attributes:

```
{
    "question_id": [string] The question id
    "source": [string] The original source of the example (for simpshort, simplong, and compshort sets)
    "original_question_id": [string] The original question id (for simpshort, simplong, and compshort sets)
    "question": [string] The question text
    "paragraphs": [list] List of paragraphs and tables within the document
    "table_evidence": [list] List of indices in 'paragraphs' that are used as table evidence for the question
    "paragraph_evidence": [list] List of indices in 'paragraphs' that are used as text evidence for the question
    "python_solution": [string] Python-format and executable solution. This feature is hidden for the test set
    "ground_truth": [float] Executed result of 'python_solution'. This feature is hidden for the test set
}
```

## Experiments
### Environment Setup
The code is tested on the following environment:
- run `pip install -r requirements.txt` to install all the required packages

### LLM Inference on DocMath-Eval
We provide inference scripts for running various LLMs on KnowledgeMath:
- `scripts/run_api.sh` for running proprietary LLMs. Note that we developed a centralized API proxy to manage API calls from different organizations and unify them to be compatible with the OpenAI API. If you use the official API platform, you will need to make some modifications.
- `scripts/run_vllm.sh` for running all other open-sourced LLMs (e.g., Llama-3, Qwen, Gemma) that are reported in the paper and supported by the [vLLM](https://github.com/vllm-project/vllm) framework

### Automated Evaluation
We develop a heuristic-based method to automatically evaluate the accuracy of CoT and PoT outputs:
- `scripts/evaluate_all.sh` for evaluating PoT and CoT outputs

To get the results on the test set, please send your result json file to [this email](mailto:yilun.zhao@yale.edu) (see the leaderboard section below for more details).

## Contact
For any issues or questions, kindly email us at: Yilun Zhao (yilun.zhao@yale.edu).

## Citation

If you use the **DocMath-Eval** benchmark in your work, please kindly cite the paper:

```
@misc{zhao2023docmatheval,
      title={DocMath-Eval: Evaluating Numerical Reasoning Capabilities of LLMs in Understanding Long Documents with Tabular Data}, 
      author={Yilun Zhao and Yitao Long and Hongjun Liu and Linyong Nan and Lyuhao Chen and Ryo Kamoi and Yixin Liu and Xiangru Tang and Rui Zhang and Arman Cohan},
      year={2023},
      eprint={2311.09805},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.09805}, 
}
```