import argparse

from gliner import GLiNER
from gliner.evaluation import get_for_one_path,get_for_all_path
import os
import json
from seqeval.metrics.sequence_labeling import get_entities

import numpy as np


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.floating):
        return float(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, np.bool_):
        return bool(obj)

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

ent_map = {
    "PRODUCT" : "sản phẩm",
    "PRODUCT FEATURE" : "đặc trưng sản phẩm",
    "PRODUCT USAGE" : "công dụng sản phẩm",
    "PRODUCT QUALITY" : "chất lượng sản phẩm",
    "PRODUCT DESIGN" : "thiết kế sản phẩm",
    "PRICE" : "giá cả",
    "SERVICE" : "dịch vụ",
    "BRANDING" : "thương hiệu",
    "GENERAL" : "chung",
    "DELIVERY" : "giao hàng"
}
def read_conll_file(file_path, ent_path = None):
    data = []
    
    tokens = []
    tags = []

    if ent_path:
        try:
            with open(ent_path, "r", encoding="utf-8") as f:
                list_ent = [line.strip() for line in f if line.strip()]
                list_ent = [ent_map[ent] for ent in list_ent]
        except:
            list_ent = None
    else:
        list_ent = None
    
    if "train" not in file_path:
        list_ent = None
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # gặp dòng trống => kết thúc 1 sample
            if not line:
                if tokens:
                    data.append(convert_to_span(tokens, tags, list_ent))
                    tokens, tags = [], []
                continue
            
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            
            token, tag = parts
            tokens.append(token)
            tags.append(tag)
    
    # case file không kết thúc bằng dòng trống
    if tokens:
        data.append(convert_to_span(tokens, tags, list_ent))
    
    return data

def convert_to_span(tokens, tags, list_ent = None):
    entities = get_entities(tags)
    new_tags = []
    for tag in tags:
        if tag == "O":
            new_tags.append(tag)
            continue
        splits = tag.split("-")
        new_tags.append(splits[0]+"-"+ent_map[splits[1]])
    
    ner = []
    ner_labels = []

    for ent_type, start, end in entities:
        ent_type = ent_map[ent_type]
        ner.append([start,end, ent_type])
        ner_labels.append(ent_type)
    if list_ent:
        ner_labels = set(ner_labels)
        ner_negatives = set(list_ent) - ner_labels

        return{
            "tokenized_text": tokens,
            "ner": ner,
            "ner_labels": list(ner_labels),
            "ner_negatives": list(ner_negatives),
            "tags" : new_tags
        }
        

    return {
        "tokenized_text": tokens,
        "ner": ner,
        "tags" : new_tags
    }

def get_best_model_checkpoint(base_path):
    """
    base_path ví dụ:
    output/phobert_large
    """

    # tìm tất cả folder checkpoint-*
    checkpoints = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
        and d.startswith("checkpoint-")
    ]

    if not checkpoints:
        raise ValueError(f"Không tìm thấy checkpoint nào trong: {base_path}")

    # sắp xếp theo số checkpoint để lấy mới nhất
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = checkpoints[-1]

    print("Latest checkpoint:", latest_checkpoint)

    # đường dẫn tới trainer_state.json
    trainer_state_path = os.path.join(
        latest_checkpoint,
        "trainer_state.json"
    )

    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(
            f"Không tìm thấy trainer_state.json tại: {trainer_state_path}"
        )

    # đọc trainer_state.json
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        trainer_state = json.load(f)

    best_model_checkpoint = trainer_state.get("best_model_checkpoint")

    if not best_model_checkpoint:
        raise ValueError(
            "Không tìm thấy best_model_checkpoint trong trainer_state.json"
        )

    return best_model_checkpoint


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--model", type=str, default="/kaggle/working/GLiNER/models/checkpoint-15950", help="Path to model folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default =0.5)
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to model folder")
    parser.add_argument('--data', type=str, default='valid_data', help='Path to the eval datasets directory')
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

    best_model_checkpoint = get_best_model_checkpoint(args.model)

    model = GLiNER.from_pretrained(best_model_checkpoint, load_tokenizer=True).to("cuda:0")
    all_results = {}
    total_f1 = 0
    for folder in subfolders:
        path = os.path.join(args.data, folder)
        data_name, results, f1, preds = get_for_one_path(path, model, threshold=args.threshold, return_preds=True, batch_size=args.batch_size)
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
    os.makedirs(args.log_dir, exist_ok=True)
    full_file_path =f"{args.log_dir}/results-{file_name}_threshold_{args.threshold}.json"
    with open( full_file_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)


        # =========================================================
    # PHẦN CUỐI FILE: thêm metrics classification vào all_results
    # =========================================================

    gt = read_conll_file("custom_train_data/v3.4/test.txt")

    with open(full_file_path, "r", encoding="utf-8") as f:
        prediction = json.load(f)
        preds = prediction["my_data"]["predictions"]


    # ---------------------------------------------------------
    # build y_true / y_pred
    # ---------------------------------------------------------
    y_true = []
    y_pred = []

    for i in range(len(gt)):
        tokens = gt[i]["tokenized_text"]
        pred_tag = ["O"] * len(tokens)

        for pred in preds[i]:
            pred_tag[pred["start"]] = "B-" + pred["label"]

            I = ["I-" + pred["label"]] * (pred["end"] - pred["start"])
            pred_tag[pred["start"] + 1 : pred["end"] + 1] = I

        y_true.append(gt[i]["tags"])
        y_pred.append(pred_tag)


    # =========================================================
    # CLASSIFICATION REPORTS
    # =========================================================

    from seqeval.metrics import classification_report as seqeval_report
    from sklearn.metrics import classification_report as sklearn_report


    # ---------------------------------------------------------
    # 1. entity-level report (seqeval)
    # ---------------------------------------------------------
    entity_report_dict = seqeval_report(
        y_true,
        y_pred,
        zero_division=0,
        digits=4,
        output_dict=True
    )

    print("========== ENTITY LEVEL REPORT ==========")
    print(seqeval_report(
        y_true,
        y_pred,
        zero_division=0,
        digits=4
    ))

    all_results["entity_level_report"] = entity_report_dict


    # ---------------------------------------------------------
    # 2. token-level report (sklearn)
    # ---------------------------------------------------------
    y_true_flat = [tag for sent in y_true for tag in sent]
    y_pred_flat = [tag for sent in y_pred for tag in sent]

    token_report_dict = sklearn_report(
        y_true_flat,
        y_pred_flat,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    print("========== TOKEN LEVEL REPORT ==========")
    print(sklearn_report(
        y_true_flat,
        y_pred_flat,
        digits=4,
        zero_division=0
    ))

    all_results["token_level_report"] = token_report_dict


    # ---------------------------------------------------------
    # 3. chỉ xét B/I/O (bỏ label type)
    # ---------------------------------------------------------
    y_t = [tag.split("-")[0] for tag in y_true_flat]
    y_p = [tag.split("-")[0] for tag in y_pred_flat]

    bio_report_dict = sklearn_report(
        y_t,
        y_p,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    print("========== BIO ONLY REPORT ==========")
    print(sklearn_report(
        y_t,
        y_p,
        digits=4,
        zero_division=0
    ))

    all_results["bio_only_report"] = bio_report_dict


    # ---------------------------------------------------------
    # 4. span-only report
    # B-SPAN / I-SPAN / O
    # ---------------------------------------------------------
    y_span_t = []
    y_span_p = []

    for sent in y_true:
        tags = []
        for tag in sent:
            if tag != "O":
                tag = tag[:2] + "SPAN"
            tags.append(tag)
        y_span_t.append(tags)

    for sent in y_pred:
        tags = []
        for tag in sent:
            if tag != "O":
                tag = tag[:2] + "SPAN"
            tags.append(tag)
        y_span_p.append(tags)

    span_report_dict = seqeval_report(
        y_span_t,
        y_span_p,
        zero_division=0,
        digits=4,
        output_dict=True
    )

    print("========== SPAN ONLY REPORT ==========")
    print(seqeval_report(
        y_span_t,
        y_span_p,
        zero_division=0,
        digits=4
    ))

    all_results["span_only_report"] = span_report_dict


    # =========================================================
    # GHI LẠI TOÀN BỘ all_results vào file JSON
    # =========================================================

    with open(full_file_path, "w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=4,
            default=convert_numpy
        )

    print(f"Saved full results with metrics to: {full_file_path}")


 
