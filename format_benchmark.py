from seqeval.metrics.sequence_labeling import get_entities

def read_conll_file(file_path, ent_path):
    data = []
    
    tokens = []
    tags = []

    if ent_path:
        try:
            with open(ent_path, "r", encoding="utf-8") as f:
                list_ent = [line.strip().lower() for line in f if line.strip()]
        except:
            list_ent = None
    else:
        list_ent = None
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # gặp dòng trống => kết thúc 1 sample
            if not line:
                if tokens:
                    data.append(convert_to_span(tokens, tags))
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
        data.append(convert_to_span(tokens, tags))
    
    return data

def convert_to_span(tokens, tags):
    entities = get_entities(tags)
    
    ner = []
    for ent_type, start, end in entities:
        ner.append(
            {
                "pos" : (start, end),
                "type": ent_type.lower()
            }
        )
    
    sentence = " ".join(tokens)

    return {
        "sentence": sentence,
        "entities": ner
    }

import os
import json

def process_folder(folder_path):
    label_path = os.path.join(folder_path, "label.txt")
    with open(label_path, "r", encoding="utf-8") as f:
        list_ent = [line.strip().lower() for line in f if line.strip()]
    
    with open(label_path.replace(".txt", "s.json"), "w", encoding="utf-8") as f:
        json.dump(list_ent, f, ensure_ascii=False, indent=4)
    
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
        print(f"Saved: {json_name}")
        print(f"Num Samples : {len(data)}")
        print("---"*15)

if __name__ == "__main__":
    path = "/kaggle/working/GLiNER/my_benchmark/"

    subfolders = [
        f for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f))
    ]

    print(f"Found {len(subfolders)} dataset.")
    for folder in subfolders:
        print("---", folder)
    # process_folder("/kaggle/working/GLiNER/my_benchmark/mit_restaurant")
    for folder in subfolders:
        print("---"*10, folder, "---"*10)
        dataset_path = os.path.join(path, folder)
        process_folder(dataset_path)
        