import json
import random
import os

def shuffle_data_files(*file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            file_data = [json.loads(line) for line in f]
            data.append(file_data)
    indices = list(range(len(data[0])))

    random.shuffle(indices)

    shuffled_data = [[] for _ in range(len(data))]

    for index in indices:
        for j, file_data in enumerate(data):
            shuffled_data[j].append(file_data[index])

    for i, file_path in enumerate(file_paths):
        directory, filename = os.path.split(file_path)
        shuffled_filename = f"shuffled_{filename}"
        with open(os.path.join(directory, shuffled_filename), 'w') as f:
            for item in shuffled_data[i]:
                json.dump(item, f, indent=4)
                f.write('\n')

file_paths = ['../datasets/anchor_dataset/train/metadata_complete.jsonl', '../datasets/positive_dataset/train/metadata_complete.jsonl', '../datasets/negative_dataset/train/metadata_complete.jsonl']

shuffle_data_files(*file_paths)