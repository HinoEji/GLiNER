"""Truncate Dữ liệu"""

def read_conll(filepath):
    sentences = []

    tokens = []
    labels = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if not line:
                if tokens:
                    sentences.append((tokens, labels))
                tokens = []
                labels = []
                continue

            parts = line.split()

            token = parts[0]
            label = " ".join(parts[1:])

            tokens.append(token)
            labels.append(label)

    if tokens:
        sentences.append((tokens, labels))

    return sentences


def truncate_sentences(sentences, max_length):
    truncated = []

    for tokens, labels in sentences:
        truncated.append(
            (
                tokens[:max_length],
                labels[:max_length]
            )
        )

    return truncated


def save_conll(sentences, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for tokens, labels in sentences:
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")

if __name__ == "__main__":

    MAX_LENGTH = 200

    input_files = [
        "custom_train_data/v3.4_origin/train.txt",
        "custom_train_data/v3.4_origin/dev.txt",
        "custom_train_data/v3.4_origin/test.txt"
    ]

    output_files = [
        "custom_train_data/v3.4_truncate/train.txt",
        "custom_train_data/v3.4_truncate/dev.txt",
        "custom_train_data/v3.4_truncate/test.txt"
    ]

    for input_file, output_file in zip(input_files, output_files):
        sentences = read_conll(input_file)
        truncated_sentences = truncate_sentences(sentences, MAX_LENGTH)
        save_conll(truncated_sentences, output_file)

        print(f"Truncated sentences saved to {output_file}")


