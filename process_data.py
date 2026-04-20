"""
read from file csv
remove sample > 200 token (words)
write to txt file
"""

import pandas as pd
import ast
from typing import List, Dict, Any, Tuple
import random
import json

def process_csv(path):
    """
    convert csv to list of dict with format:
    {
        "tokens": ["token1", "token2", ...],
        "tags": ["tag1", "tag2", ...]
    }
    check if the token and tag length is equal
    remove sample > 200 token (words) (for phobert)
    """
    new_data = []
    df = pd.read_csv(path)
    for row in df.itertuples():
        index = row.Index_sentence
        data = ast.literal_eval(row.sc_label)
        text = data['text'].strip()
        tokens = text.split()
        tags = data['SC']
        if len(tokens) != len(tags):
            print(f"[WARNING]: {index} - len num tokens ({len(tokens)}) != num tags ({len(tags)})")
        if len(tokens) > 200:
            print(f"[SKIP]: {index} - len num tokens ({len(tokens)}) > 200")
            continue
        new_data.append({
            "id": index,
            "tokens": tokens,
            "tags": tags
        })
    return new_data

    

def convert_to_conll_format(data, output_file):
    """
    list of data in format:
    {
        "tokens": ["token1", "token2", ...],
        "tags": ["tag1", "tag2", ...]
    }
    return 
    """
    lines = []

    for sample in data:
        tokens = sample["tokens"]
        tags = sample["tags"]

        assert len(tokens) == len(tags)

        for token, tag in zip(tokens, tags):
            lines.append(f"{token}\t{tag}")

        lines.append("")

    conll_str = "\n".join(lines)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(conll_str)

    return conll_str



def split_data(entries: List[Dict[str, Any]], 
               train_ratio: float = 0.8,
               test_ratio: float = 0.1,
               dev_ratio: float = 0.1,
               seed: int = 42) -> Tuple[List, List, List]:
    """
    Returns:
        Tuple of (train_data, test_data, dev_data)
    """
    random.seed(seed)
    random.shuffle(entries)
    
    total = len(entries)
    train_size = int(total * train_ratio)
    test_size = int(total * test_ratio)
    
    train_data = entries[:train_size]
    test_data = entries[train_size:train_size + test_size]
    dev_data = entries[train_size + test_size:]
    
    return train_data, test_data, dev_data


if __name__ == "__main__":
    data_path = "/kaggle/working/GLiNER/custom_train_data/v3.4/data_v3.4.csv"
    train_path = "/kaggle/working/GLiNER/custom_train_data/v3.4/train.txt"
    test_path = "/kaggle/working/GLiNER/custom_train_data/v3.4/test.txt"
    dev_path = "/kaggle/working/GLiNER/custom_train_data/v3.4/dev.txt"
    
    data = process_csv(data_path)
    with open(data_path.replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"[INFO]: {len(data)} samples")
    train_data, test_data, dev_data = split_data(data)
    print(f"[INFO]: {len(train_data)} train samples")
    print(f"[INFO]: {len(test_data)} test samples")
    print(f"[INFO]: {len(dev_data)} dev samples")
    convert_to_conll_format(train_data, train_path)
    convert_to_conll_format(test_data, test_path)
    convert_to_conll_format(dev_data, dev_path)