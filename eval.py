import argparse

from gliner import GLiNER
from gliner.evaluation import get_for_one_path,get_for_all_path
import os
import json


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--model", type=str, default="HinoEiji/GLiNER-phobert-large", help="Path to model folder")
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
        data_name, results, f1, preds = get_for_one_path(path, model, threshold=threshold, return_preds=True)
        print("============="*5, data_name, "============="*5)
        
        # Serialize predictions for JSON dumping
        serialized_preds = []
        for p_list in preds:
            cur_list = []
            for p in p_list:
                if hasattr(p, 'start'):
                    cur_list.append({
                        "start": p.start, 
                        "end": p.end, 
                        "label": getattr(p, 'entity_type', getattr(p, 'label', 'UNK')),
                        "score": float(p.score) if hasattr(p, 'score') else None
                    })
                elif isinstance(p, dict):
                    cur_list.append(p)
                elif isinstance(p, (list, tuple)) and len(p) >= 3:
                    cur_list.append({"start": p[0], "end": p[1], "label": p[2]})
                else:
                    cur_list.append(str(p))
            serialized_preds.append(cur_list)
            
        all_results[data_name] = {}
        all_results[data_name]['result'] = results
        all_results[data_name]['f1'] = f1
        all_results[data_name]['predictions'] = serialized_preds
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
 
