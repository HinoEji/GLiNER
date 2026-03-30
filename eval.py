import argparse

from gliner import GLiNER
from gliner.evaluation import get_for_one_path,get_for_all_path
import os
import json


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--model", type=str, default="/kaggle/working/GLiNER/models/checkpoint-17864", help="Path to model folder")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to model folder")
    parser.add_argument('--data', type=str, default='/kaggle/working/GLiNER/valid_data', help='Path to the eval datasets directory')
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    subfolders = [
        f for f in os.listdir(args.data)
        if os.path.isdir(os.path.join(args.data, f))
    ]
    print(f"Found {len(subfolders)} dataset.")
    for folder in subfolders:
        print("---", folder)

    model = GLiNER.from_pretrained(args.model, load_tokenizer=True).to("cuda:0")
    threshold = 0.9
    all_results = {}
    total_f1 = 0
    for folder in subfolders:
        path = os.path.join(args.data, folder)
        data_name, results, f1 = get_for_one_path(path, model, threshold=threshold)
        print("============="*5, data_name, "============="*5)
        all_results[data_name] = {}
        all_results[data_name]['result'] = results
        all_results[data_name]['f1'] = f1
        print(all_results[data_name]['result'])
        total_f1 += f1
    

    for key, value in all_results.items():
        score = value['f1']
        print(f"{key} : {score}" )
    all_results['avg_f1'] = total_f1 / len(subfolders)
    print("Avg F1 : ", all_results['avg_f1'])
    
    file_name = args.model.split("/")[-1]
    with open(f"results-{file_name}_threshold_{threshold}.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
 
