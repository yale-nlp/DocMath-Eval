import json, os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    args = parser.parse_args()

    data = json.load(open(args.input_file))

    top_k_data_sorted_by_ids = []

    for entry in data:
        sorted_paragraphs_by_scores = sorted(entry["retrieved_paragraphs"], key=lambda x: x[1], reverse=True)
        top_k_paragraphs = sorted_paragraphs_by_scores[:args.top_k]
        top_k_paragraphs_sorted_by_ids = sorted(top_k_paragraphs, key=lambda x: x[0])
        
        top_k_entry = {
            "question_id": entry["question_id"],
            "retrieved_paragraphs": top_k_paragraphs_sorted_by_ids
        }
        top_k_data_sorted_by_ids.append(top_k_entry)

    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, os.path.basename(args.input_file))

    json.dump(top_k_data_sorted_by_ids, open(output_file, "w"), indent=4)
