import random

DATA_FILE_PATH = "../models/Yelp/generatedyelp.txt"
OUT_FILE_PATH = "../models/Yelp/consistency_manual_labeled_dataset.txt"

with open(DATA_FILE_PATH, "r") as f:
    data_to_label = f.readlines()

data_to_label = [line.strip() for line in data_to_label]
results = []
sample_indices = list(range(0, len(data_to_label), 2))
random.shuffle(sample_indices)

while (answer := input(f"{data_to_label[sample_indices[len(results)]]}\nis pair consistent? [q]uit\n")) != "q":
    if answer in {"0", "1"}:
        results.append(f"{sample_indices[len(results)] + 1}\t{data_to_label[sample_indices[len(results)]]}\t{answer}")

with open(OUT_FILE_PATH, "a") as f:
    f.write("\n".join(results))