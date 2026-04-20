from seqeval.metrics.sequence_labeling import get_entities
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
            "ner_negatives": list(ner_negatives)
        }
        

    return {
        "tokenized_text": tokens,
        "ner": ner
    }

import os
import json

def process_folder(folder_path):
    label_path = os.path.join(folder_path, "label.txt")
    
    for file_name in os.listdir(folder_path):
        # bỏ qua label.txt
        if file_name == "label.txt":
            continue
        
        # chỉ xử lý file .txt
        if not file_name.endswith(".txt"):
            continue
        
        file_path = os.path.join(folder_path, file_name)
        
        print(f"Processing: {file_name}")
        
        # đọc và convert
        data = read_conll_file(file_path, ent_path=label_path)
        
        # tạo tên file json
        json_name = file_name.replace(".txt", ".json")
        json_path = os.path.join(folder_path, json_name)
        
        # save
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("---"*15)
        print(f"Saved: {json_name}")
        print(f"Num Samples : {len(data)}")
        print("---"*15)
        with open(label_path, "r", encoding="utf-8") as f:
            list_ent = [line.strip() for line in f if line.strip()]
            list_ent = [ent_map[ent] for ent in list_ent]
        with open(label_path.replace(".txt", "s.json"), "w", encoding="utf-8") as f:
            json.dump(list_ent, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    process_folder("/kaggle/working/GLiNER/custom_train_data/v3.4")