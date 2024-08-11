import os, json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--ground_truth_file", type=str, required=True)
    args = parser.parse_args()

    data = json.load(open(args.input_file))
    ground_truth_data = json.load(open(args.ground_truth_file))

    gt_dict = {}
    for example in ground_truth_data:
        question_id = example["question_id"]
        gt_dict[question_id] = set(example["table_evidence"] + example["paragraph_evidence"])

    recalls = []
    for example in data:
        question_id = example["question_id"]
        retrieved_paragraphs = example["retrieved_paragraphs"]
        retrieved_paragraphs_ids = [x[0] for x in retrieved_paragraphs]

        matched = gt_dict[question_id].intersection(set(retrieved_paragraphs_ids))
        recalls.append(len(matched) / len(gt_dict[question_id]))

    recall = round(sum(recalls) / len(recalls) * 100, 2)


    file_name = os.path.basename(args.input_file)
    top_k = os.path.basename(os.path.dirname(args.input_file))
    output_dir = os.path.dirname(os.path.dirname(args.input_file))
    output_file = os.path.join(output_dir, f"results_{top_k}.json")
    
    if os.path.exists(output_file):
        results = json.load(open(output_file))
    else:
        results = {}

    results[file_name] = recall
    json.dump(results, open(output_file, "w"), indent=4)

    print(f"For {args.input_file}, recall is {recall}")
        
        
